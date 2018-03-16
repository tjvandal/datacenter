import xarray as xr
import numpy as np
import os, sys
import sr_utils
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

import matplotlib.pyplot as plt

'''
data/rcp85/MRI-CGCM3/pr/pr_day_MRI-CGCM3_rcp85_r1i1p1_20060101-20151231.nc
1. Download necessary data
2. Given a model, scenario, and variables return GCMs at 1.3333 degrees spatial resolution
'''

class GCMBase(object):
    '''
    Reads a single GCM file
    '''
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = xr.open_dataset(self.datafile)
        self.data.lon.values = self.data.lon.values % 360


class GCMGroup(object):
    '''
    Joins multiple GCMs into a single object
    Input:
        gcms: List of GCMBase objects
    '''
    def __init__(self, gcms_files):
        self.gcms = gcms


class GCMDeepSD(object):
    def __init__(self, gcm_files, elevation_file=os.path.join(FILE_DIR, 'wcs_4km_prism.nc'),
                variables=['pr', 'tasmax', 'tasmin'], low_resolution=4/3., n_stacked=5):
        self.gcm_files = gcm_files
        self.gcms = xr.open_mfdataset(self.gcm_files)
        self.variables = variables
        if 'pr' in self.variables:
            self.gcms['pr'] = self.gcms['pr']*3600*24   #kg/m^2/s to mm/day
        if 'tasmax' in self.variables:
            self.gcms['tasmax'] = self.gcms['tasmax'] - 273.15 # kelvin to celsius
        if 'tasmin' in self.variables:
            self.gcms['tasmin'] = self.gcms['tasmin'] - 273.15 # kelvin to celsius
        self.low_resolution = low_resolution
        self.upscale_factor = 2.
        self.n_stacked = n_stacked
        self.elevation = self.read_elevation(elevation_file)

    def read_elevation(self, file):
        self.elev = xr.open_dataset(file)['Band1']
        self.elev.lon.values = self.elev.lon.values % 360
        h, w = self.elev.shape
        s = int(self.upscale_factor**self.n_stacked)
        ch = h  - h % s
        cw = w - w % s
        self.elev = self.elev.isel(lat=slice(0, ch), lon=slice(0,cw))
        self.lr_elev = sr_utils.interp_da2d(self.elev, scale=(1./s))
        self._regrid_gcms()

    def _regrid_latlon(self, step):
        ltmin = self.gcms.lat.min()
        lnmin = self.gcms.lon.min()
        ltmax = self.gcms.lat.max()
        lnmax = self.gcms.lon.max()
        latnew = np.arange(ltmin, ltmax, step=step)
        lonnew = np.arange(lnmin, lnmax, step=step)
        return latnew, lonnew

    def _regrid_gcms(self):
        # select pixels closest to self.lr_elev
        lts = self.lr_elev.lat
        lns = self.lr_elev.lon

        latnew, lonnew = self._regrid_latlon(self.low_resolution)
        ltmin = np.min(latnew)
        ltmax = np.max(latnew)
        lnmin = np.min(lonnew)
        lnmax = np.max(lonnew)

        newshape = (len(latnew), len(lonnew))
        newds = sr_utils.interp_ds(self.gcms, self.variables, newshape=newshape)
        newds.lon.values = lonnew
        newds.lat.values = latnew
        self.gcms = newds.sel(lat=lts, lon=lns, method='nearest')

    def _regrid_elevs(self):
        '''
        Returns elevations for each level of downscaling
        ie. Deepsd with 5 networks needs 5 elevation inputs with increasing resolutions
        '''
        ds0 = self.gcm_conus.isel(time=[0,1])
        nlat = ds0.lat.shape[0]
        nlon = ds0.lon.shape[0]
        for n in range(1, self.n_stacked+1):
            r = self.upscale_factor**n
            deg = 1. * self.low_resolution / r
            newshape = (int(nlat * r),int(nlon*r))
            newds = sr_utils.interp_ds(ds0, self.variables, newshape=newshape)
            #print self._regrid_conus(ds0, deg)

    def get_elev_stack(self):
        elevs = []
        for i in range(self.n_stacked):
            r = int(self.upscale_factor**i)
            elevs.append(sr_utils.interp_da2d(self.elev, scale=1./r).values)
        return elevs[::-1]

    def get_graph_inputs(self):
        '''
        Steps:
            1. Interpolate to 1 1/3 degrees
            2. Box CONUS
            3. Put elevation(s) on the same grid as GCM
        Attempt 2:
            1. Upscale Elevation to low-resolution
            2. Put GCM on grid of elevation
        '''
        elev_stack = self.get_elev_stack()
        #self.gcm_conus = self._regrid_conus(self.gcms, self.low_resolution)
        X = [self.gcms[v].values[:,:,:,np.newaxis] for v in self.variables]
        X = np.concatenate(X, axis=3)
        return X, elev_stack


class FileStructure(object):
    def __init__(self, base_dir=os.path.join(FILE_DIR, 'data')):
        self.base_dir = base_dir
        print 'basedir', base_dir

    def scenarios(self):
        return os.listdir(self.base_dir)

    def _models(self, s):
        return os.listdir('%s/%s' % (self.base_dir, s))

    def _variables(self, scenario, model):
        return os.listdir('%s/%s/%s/' % (self.base_dir, scenario, model))

    def _variable_files(self, scenario, model, variable):
        path = '%s/%s/%s/%s/' % (self.base_dir, scenario, model, variable)
        return sorted([os.path.join(path, f) for f in os.listdir(path)])

    def get_scenario_model_files(self, scenario, model, variables=None):
        '''
        Args:
            variables: list of variables pr,tasmax,tasmin
        '''
        if variables is None:
            variables = self._variables(scenario, model)
        files = dict()
        for v in variables:
            files[v] = self._variable_files(scenario, model, v)
        return files

    def walk_scenarios_models(self, variables):
        for s in self.scenarios():
            for model in self._models(s):
                d = self.get_scenario_model_files(s, model, variables=variables)
                n_files = len(d[variables[0]])
                for i in range(n_files):
                    group = [d[v][i] for v in variables]
                    yield s, model, group

if __name__ == "__main__":
    struct = FileStructure()
    for s, m, filegroup in struct.walk_scenarios_models(['pr', 'tasmax', 'tasmin']):
        gcminputs = GCMDeepSD(filegroup)
        sys.exit()

