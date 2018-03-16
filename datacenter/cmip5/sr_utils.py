import xarray as xr
import numpy as np
import cv2
from scipy.misc import imresize
import scipy.interpolate

def fillmiss(x):
    if x.ndim != 2:
        raise ValueError("X have only 2 dimensions.")
    mask = ~np.isnan(x)
    xx, yy = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    data0 = np.ravel(x[mask])
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0

def interp_dim(x, scale=None, n=None):
    assert (n is not None) or (scale is not None)
    if (n is None) and (scale is not None):
        n = int(len(x)*scale)
    x0, xlast = x[0], x[-1]
    y = np.linspace(x0, xlast, n)
    assert len(y) == n
    return y

def interp_tensor(X, scale=None, newshape=None, fill=True, how=cv2.INTER_LINEAR):
    if newshape is None:
        nlt = int(X.shape[1]*scale)
        nln = int(X.shape[2]*scale)
        newshape = (nlt, nln)
    scaled_tensor = np.empty((X.shape[0], newshape[0], newshape[1]))
    for j, im in enumerate(X):
        # fill im with nearest neighbor
        if fill:
            #im = fillmiss(im)
            im[np.isnan(im)] = 0

        scaled_tensor[j] = cv2.resize(im, (newshape[1], newshape[0]),
                                     interpolation=how)
    return scaled_tensor


def interp_da2d(da, scale=None, newshape=None, fillna=False, how=cv2.INTER_LINEAR):
    """
    Assume da is of dimensions ('time','lat', 'lon')
    """
    im = da.values
    # lets store our interpolated data
    if (newshape is None) and (scale is not None):
        newshape = (int(im.shape[0]*scale), int(im.shape[1]*scale))
    scaled_tensor = np.empty(newshape)
    # fill im with nearest neighbor
    if fillna:
        filled = fillmiss(im)
    else:
        filled = im
    scaled_tensor = cv2.resize(filled, dsize=(newshape[1], newshape[0]), interpolation=how)
    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[0]].values, n=newshape[0])
    lonnew = interp_dim(da[da.dims[1]].values, n=newshape[1])
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[latnew, lonnew],
                 dims=da.dims)

def interp_da(da, scale=None, newshape=None, how=cv2.INTER_LINEAR):
    """
    Assume da is of dimensions ('time','lat', 'lon')
    """
    if (newshape is None) and (scale is not None):
        newshape = (int(da.shape[0]*scale),int(da.shape[1]*scale))
    tensor = da.values

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[1]].values, n=newshape[0])
    lonnew = interp_dim(da[da.dims[2]].values, n=newshape[1])

    # lets store our interpolated data
    scaled_tensor = interp_tensor(tensor, newshape=newshape, fill=True, how=how)

    if latnew.shape[0] != scaled_tensor.shape[1]:
        raise ValueError("New shape is shitty")
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[da[da.dims[0]].values, latnew, lonnew],
                 dims=da.dims)

def interp_ds(ds, variables, scale=None, newshape=None, how=cv2.INTER_LINEAR):
    data = dict()
    for v in variables:
        data[v] = interp_da(ds[v], newshape=newshape, how=how)
    return xr.Dataset(data)

#def regrid_da(da, lats, lons):


if __name__=="__main__":
    import matplotlib.pyplot as plt

    fhigh = '/raid/prism/ppt_0.125x0.125/prism_ppt_interp_1981.nc'
    var='ppt'

    dshigh = xr.open_dataset(fhigh)
    dshigh = dshigh.isel(time=[0,1])
    #dshigh['ppt'] = dshigh.ppt.fillna(0)
    dalow = interp_da(dshigh.ppt, 1./8)
    danew = interp_da(dalow, 8.)

    plt.figure(figsize=(8,20))
    plt.subplot(3,1,1)
    danew.isel(time=0).plot()
    plt.subplot(3,1,2)
    dshigh.isel(time=0).ppt.plot()
    plt.subplot(3,1,3)
    dalow.isel(time=0).plot()
    plt.show()
