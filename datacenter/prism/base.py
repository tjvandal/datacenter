import os
import shutil
import sys
from datetime import datetime
import csv
import tempfile
import gdal
import gdalconst
import zipfile as zf
import numpy as np
import pandas as pd
#from unitconversion import *
from StringIO import StringIO
import xarray as xr
import ConfigParser
import sr_utils

import tensorflow as tf
from tfwriter import convert_to_tf
from tfreader import inputs_climate

def recursive_mkdir(path):
    split_dir = path.split("/")
    for k in range(len(split_dir)):
        d = "/".join(split_dir[:(k+1)])
        if (d != '') and (not os.path.exists(d)):
            os.mkdir(d)

class PrismBil:
    def __init__(self, zip_file_pointer):
        self.zf = zip_file_pointer
        self.bilname = [n for n in self.zf.namelist() if n[-4:] == '.bil'][0]
        self.hdrname = [n for n in self.zf.namelist() if n[-4:] == '.hdr'][0]
        date = self.bilname.split("_")[-2]
        self.date = pd.to_datetime(date, format='%Y%m%d')
        print self.date

    def save_temp(self):
        tmp_dir = tempfile.mkdtemp()
        print "making temp dir", tmp_dir
        for f in [self.bilname, self.hdrname]:
            fn = os.path.join(tmp_dir, f)
            with open(fn, 'wb') as file:
                file.write(self.zf.read(f))
        self.bilfile = os.path.join(tmp_dir, self.bilname)
        self.hdrfile = os.path.join(tmp_dir, self.hdrname)
        return tmp_dir

    def bil_to_xray(self):
        try:
            tmp_dir = self.save_temp()
            img = gdal.Open(self.bilfile, gdalconst.GA_ReadOnly)
            band = img.GetRasterBand(1)
            self.nodatavalue = band.GetNoDataValue()
            self.ncol = img.RasterXSize
            self.nrow = img.RasterYSize
            geotransform = img.GetGeoTransform()
            self.originX = geotransform[0]
            self.originY = geotransform[3]
            self.pixelWidth = geotransform[1]
            self.pixelHeight = geotransform[5]
            self.data = band.ReadAsArray()
            self.data = np.ma.masked_where(self.data==self.nodatavalue, self.data)
            lats = np.linspace(self.originY, self.originY + self.pixelHeight *(self.nrow - 1),
                             self.nrow)
            lons = np.linspace(self.originX, self.originX + self.pixelWidth * (self.ncol -1),
                             self.ncol)
            #print lats[:-1] - lats[1:]
            #sys.exit()
            dr = xr.DataArray(self.data[np.newaxis, :, :],
                              coords=dict(time=[self.date], lat=lats, lon=lons), 
                              dims=['time', 'lat', 'lon']) 
        finally:
            shutil.rmtree(tmp_dir)
        return dr

    def plot(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.data[:200, :200])
        plt.colorbar()
        plt.show()

    def close(self):
        self.zf.close()

def downloadPrismFtpData(parm, output_dir=os.getcwd(), timestep='monthly', years=None, server='prism.oregonstate.edu'):
    """
    Downloads ESRI BIL (.hdr) files from the PRISM FTP site.
    'parm' is the parameter of interest: 'ppt', precipitation; 'tmax', temperature, max' 'tmin', temperature, min /
                                         'tmean', temperature, mean
    'timestep' is either 'monthly' or 'daily'. This string is used to direct the function to the right set of remote folders.
    'years' is a list of the years for which data is desired.
    """
    from ftplib import FTP
    import socket
    if type(years) == int:
        years = [years]

    recursive_mkdir(output_dir)
    data = []

    def handleDownload(block):
        data.append(block)

    # Play some defense
    assert parm in ['ppt', 'tmax', 'tmean', 'tmin'], "'parm' must be one of: ['ppt', 'tmax', 'tmean', 'tmin']"
    assert timestep in ['daily', 'monthly'], "'timestep' must be one of: ['daily', 'monthly']"
    assert years is not None, 'Please enter a year for which data will be fetched.'
    if isinstance(years, int):
        years = list(years)
    try:
        ftp = FTP(server, timeout=5)
        ftp.login()
    except socket.timeout:
        print("Cannot connect to FTP server, socket.timeout")
        return
    # Wrap everything in a try clause so we close the FTP connection gracefully
    try:
        for year in years:
            save_nc_file = os.path.join(output_dir, "prism_%s_4km2_%04i.nc" % (parm, year))
            if os.path.exists(save_nc_file):
                continue
            data = []
            xray_data = []
            if timestep == 'daily':
                dir = timestep
            dir_string = '{}/{}/{}'.format(dir, parm, year)
            remote_files = sorted(ftp.nlst(dir_string))
            for f_string in remote_files:
                print f_string
                f = f_string.rsplit(' ')[-1]
                if not '_bil' in f:
                    continue

                f_path = '{}'.format(f)
                ftp.retrbinary('RETR ' + f_path, handleDownload)
                c = StringIO("".join(data))
                with zf.ZipFile(c) as z:
                    p = PrismBil(z)
                    xray_data.append(p.bil_to_xray())
            ds = xr.Dataset({parm: xr.concat(xray_data, dim='time')})
            ds.to_netcdf(save_nc_file, format='NETCDF3_CLASSIC')

    except Exception as e:
        print e
    finally:
        ftp.close()

    return

class PrismBase(object):
    def __init__(self, data_dir, year, elevation_file='wcs_4km_prism.nc', var='ppt'):
        self.data_dir = data_dir
        self.base_km = 4
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.var=var
        self.year = year
        self.elevation_file = elevation_file
        self.read_data()

    def _get_year_file(self):
        print "data dir", self.data_dir, self.year
        fnames = [f for f in os.listdir(self.data_dir) if str(self.year) in f]
        if len(fnames) == 1:
            return os.path.join(self.data_dir, fnames[0])
        elif len(fnames) == 0:
            downloadPrismFtpData(self.var, output_dir=self.data_dir, timestep='daily',
                                 years=self.year)
            fnames = [f for f in os.listdir(self.data_dir) if str(self.year) in f]
            return os.path.join(self.data_dir, fnames[0])

        elif len(fnames) > 1:
            raise IndexError("Multiples files for year:%i found" % self.year)

    def _read_highres(self):
        highres_file = self._get_year_file()
        print 'highres file', highres_file
        self.highres = xr.open_dataset(highres_file)

    def _read_elevation(self):
        elev = xr.open_dataset(self.elevation_file)
        self.elev = elev.rename({"Band1": "elev"})

    def read_data(self):
        self._read_highres()
        if self.elevation_file is not None:
            self._read_elevation()
            self.elev = self.elev.elev.sel(lat=self.highres.lat, lon=self.highres.lon, method='nearest')
            self.elev['lat'] = self.highres.lat
            self.elev['lon'] = self.highres.lon

class PrismSuperRes(PrismBase):
    def __init__(self, data_dir, year,
                 elevation_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wcs_4km_prism.nc'),
                 var='ppt'):
        super(PrismSuperRes, self).__init__(data_dir, year, elevation_file, var=var)

    def resolve_data(self, lr_km=8, hr_km=4):
        """
        Interpolate the data in accordance to the scaling factors
        A scaling factor of 0.5 cuts the resolution in half.
        """
        # crop data to ensure integer upscaling factors
        scale1 = 1. * self.base_km / hr_km
        scale2 = 1. * hr_km / lr_km
        factor = 1. / (scale1*scale2)
        if int(factor) != factor:
            print "Factor =", factor
            raise ValueError("factor=lr_km/hr_km should be an integer")

        factor = int(factor)

        t, h, w = self.highres[self.var].shape
        Y = self.highres[self.var]
        Y = Y.isel(lat=range(0,h - h % factor), lon=range(0,w - w % factor))

        # get a mask
        mask = 1.*Y.notnull()
        mask /= mask

        # this fills missing values
        Y_interp = sr_utils.interp_da(Y, scale1)
        elev = self.elev.isel(lat=range(0, h - h % factor),
                              lon=range(0, w - w % factor))
        elev = sr_utils.interp_da2d(elev, scale1)
        mask = mask.sel(lat=Y_interp.lat, lon=Y_interp.lon, method='nearest')

        X = sr_utils.interp_da(Y_interp, scale2)
        return mask, X, Y_interp, elev

    def make_patches(self, save_file=None, size=50, stride=30, lr_km=8, hr_km=4):
        if size is None:
            return self.make_test(lr_km=lr_km, hr_km=hr_km)

        factor = 1.*hr_km / lr_km
        assert (size * factor) == int(size*factor)
        assert (stride * factor) == int(stride * factor)

        mask, da1, da2, elev = self.resolve_data(lr_km=lr_km, hr_km=hr_km)

        obs_lats = da2.lat.values
        obs_lons = da2.lon.values
        X = da1.values
        Y = da2.values
        print "Max X", np.max(np.abs(X))
        print "Max Y", np.max(np.abs(Y))

        print "Shape of X", X.shape

        # keep elevation flexible by returning it seperately
        elev = elev.values[:Y.shape[1],:Y.shape[2],np.newaxis]
        X = np.expand_dims(X, 3)

        labels, inputs, elevs = [], [], []
        lats, lons, times = [], [], []
        timevals = da1.time.values
        mask_vars = mask.notnull().values
        land_locs = {}
        for j, t in enumerate(timevals):
            for y in np.arange(0, Y.shape[1], stride):
                for x in np.arange(0, Y.shape[2], stride):
                    if ((y+size) > Y.shape[1]) or ((x+size) > Y.shape[2]):
                        continue

                    x_lr = int(x*factor)
                    y_lr = int(y*factor)
                    s_lr = int(size*factor)
                    x_sub = X[j, np.newaxis, y_lr:y_lr+s_lr, x_lr:x_lr+s_lr]

                    # are we over the ocean? 
                    #land_ratio = mask.notnull().values[y:y+size, x:x+size].mean()
                    if (x,y) not in land_locs.keys():
                        land_locs[(x,y)] = mask_vars[y:y+size, x:x+size].mean() > 0.25
                    if not land_locs[(x,y)]:
                        continue

                    y_sub = Y[j, np.newaxis, y:y+size, x:x+size, np.newaxis]
                    elev_sub = elev[np.newaxis,y:y+size,x:x+size,:]

                    inputs += [x_sub]
                    labels += [y_sub]
                    elevs += [elev_sub]
                    lats += [obs_lats[np.newaxis, y:y+size]]
                    lons += [obs_lons[np.newaxis, x:x+size]]
                    times += [t]

        order = range(len(inputs))
        #np.random.shuffle(order)
        self.inputs = np.concatenate(inputs, axis=0)[order]
        self.labels = np.concatenate(labels, axis=0)[order]
        elevs= np.concatenate(elevs, axis=0)[order]
        lats = np.vstack(lats)[order]
        lons = np.vstack(lons)[order]
        if save_file is not None:
            convert_to_tf(self.inputs, elevs, self.labels, lats, lons, np.array(times)[order], save_file)
        return dict(inputs=self.inputs, elevs=elevs, labels=self.labels,
                    lats=lats, lons=lons, times=np.array(times)[order])

    def make_test(self, lr_km=8, hr_km=4):
        mask, da1, da2, elev = self.resolve_data(lr_km, hr_km)
        Y = (da2.values * mask.values)[:,:,:,np.newaxis]
        X = da1.values
        elev_arr = np.empty((Y.shape[0], Y.shape[1], Y.shape[2], 1))
        elev_arr[:] = elev.values[:,:,np.newaxis]
        X = np.expand_dims(X, 3)

        times = da2.time.values
        lats = [da2.lat.values for i in range(Y.shape[0])]
        lons = [da2.lon.values for i in range(Y.shape[0])]
        return dict(inputs=X, elevs=elev_arr, labels=Y, lats=lats, lons=lons, times=times)

class PrismTFPipeline:
    def __init__(self, data_dir, years=[2014],
                 elevation_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wcs_4km_prism.nc'),
                 input_vars=['ppt'], output_vars=['ppt'],
                 lr_km=8, hr_km=4):
        self.data_dir = data_dir
        self.years = years
        self.elevation_file = elevation_file
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.all_vars = np.unique(input_vars + output_vars)
        self.lr_km = lr_km
        self.hr_km = hr_km

    def get_patches(self, y, patch_size=None, stride=None):
        year_data = {}
        for v in self.all_vars:
            curr_prism = PrismSuperRes(self.data_dir + '/' + v, y, self.elevation_file, var=v)
            if patch_size != None:
                print "making patches"
                year_data[v] = curr_prism.make_patches(size=patch_size, stride=stride,
                                                   lr_km=self.lr_km, hr_km=self.hr_km)
            else:
                year_data[v] = curr_prism.make_test(lr_km=self.lr_km, hr_km=self.hr_km)

        # lets figure out which times are available for each variable
        t = [year_data[v]['times'] for v in self.all_vars]
        t_unique = np.unique(np.concatenate(t))
        for v in year_data:
            rows = np.in1d(year_data[v]['times'], t_unique)
            for k in ['inputs', 'labels', 'times']:
                year_data[v][k] = year_data[v][k][rows]

        X = [year_data[v]['inputs'] for v in self.input_vars]
        X = np.concatenate(X, axis=3)
        Y = [year_data[v]['labels'] for v in self.output_vars]
        Y = np.concatenate(Y, axis=3)
        elevs = year_data[self.all_vars[0]]['elevs']
        t = year_data[self.all_vars[0]]['times']
        lats = year_data[self.all_vars[0]]['lats']
        lons = year_data[self.all_vars[0]]['lons']
        return X, elevs, Y, lats, lons, t

    def _save_patches(self, patch_size=None, stride=None, force=False):
        if patch_size is None:
            patch_save_dir = os.path.join(self.data_dir,"_".join(self.all_vars), 'full_image-%i_%i'
                                         % (self.lr_km, self.hr_km))
        else:
            patches = True
            patch_save_dir = os.path.join(self.data_dir,"_".join(self.all_vars), 'patch_%i-%i_%i' %
                                          (patch_size, self.lr_km, self.hr_km))
        if not os.path.exists(patch_save_dir):
            os.makedirs(patch_save_dir)

        files = []
        for y in self.years:
            patch_save_file = os.path.join(patch_save_dir, str(y) + '.tfrecords')
            files.append(patch_save_file)
            if (os.path.exists(patch_save_file)) and (not force):
                continue

            X, elevs, Y, lats, lons, t = self.get_patches(y, patch_size, stride)
            print "Shape of input", X.shape, "Shape of label", Y.shape
            convert_to_tf(X, elevs, Y, lats, lons, t, patch_save_file)
        return files

    def tf_patches(self, batch_size=20, patch_size=38, stride=20, scope=None,
                  epochs=int(1e10), is_training=True):
        """
        A pipeline for reading patch tf files
        """
        patch_files = self._save_patches(patch_size, stride)
        factor = 1.*self.hr_km / self.lr_km
        if patch_size == None:
            lr_shape = None
        else:
            lr_d = int(factor * patch_size)
            lr_shape = [lr_d, lr_d]
        with tf.variable_scope(scope, "inputs_climate_patch"):
            images, auxs, labels, t = inputs_climate(batch_size, is_training, epochs, patch_files,
                            len(self.input_vars), 1, len(self.output_vars), lr_shape=lr_shape,
                            hr_shape=[patch_size, patch_size])
        return images, auxs, labels, t

    def tf_test(self, batch_size=1, scope=None, epochs=int(1e8)):
        test_files = self._save_patches()
        with tf.variable_scope(scope, "inputs_climate_test"):
            images, auxs, labels, times = inputs_climate(batch_size, False, epochs, test_files,
                            len(self.input_vars), 1, len(self.output_vars))
        return images, auxs, labels, times

if __name__ == "__main__":
    years = [1990] # range(1981, 1986)
    p = PrismTFPipeline('/home/tj/repos/datacenter/datacenter/prism/data/',
                        years=years, lr_km=64, hr_km=16, input_vars=['ppt'])
    #print p.tf_patches(patch_size=64, batch_size=1)
    z = p.get_patches(1991, patch_size=64, stride=48)
    print z[0].shape, np.max(np.abs(z[0])), np.max(np.abs(z[2]))
