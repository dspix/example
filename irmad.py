#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:56:08 2018

@author: dan

The irmad code comes from https://github.com/mortcanty/CRCPython
"""
import gdal
import ogr
import sys
import numpy as np
import warnings
import math

from skimage.measure import block_reduce
from scipy import ndimage
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append('/home/dan/utils/safs')
#import gpixel as gp
import copy

#import auxil.auxil as auxil 
from scipy import linalg, stats

from scipy.odr import Model, Data, ODR
from scipy.stats import linregress

import platform
import ctypes
from numpy.ctypeslib import ndpointer
import os

fpath, _ = os.path.split(__file__)
print(fpath)
if platform.system() == 'Windows':
    lib = ctypes.cdll.LoadLibrary(os.path.join(fpath, 'prov_means.dll'))
elif platform.system() == 'Linux':
    lib = ctypes.cdll.LoadLibrary(os.path.join(fpath,'libprov_means.so'))
elif platform.system() == 'Darwin':
    lib = ctypes.cdll.LoadLibrary(os.path.join(fpath,'libprov_means.dylib'))
provmeans = lib.provmeans
provmeans.restype = None
c_double_p = ctypes.POINTER(ctypes.c_double)
provmeans.argtypes = [
    ndpointer(np.float64),
    ndpointer(np.float64),
    ctypes.c_int,
    ctypes.c_int,
    c_double_p,
    ndpointer(np.float64),
    ndpointer(np.float64)
]

def f(B, x):
    '''Linear function y = m*x + b'''
    return B[0]*x + B[1]

class orthogonal_linear_model():
    
    def __init__(self):
        self.betas = [0., 1.]
        self.model = f

    def fit(self, x, y):
        # Initial estimate of betas
        linreg = linregress(x, y)

        linear = Model(self.model)
        mydata = Data(x, y)
        myodr = ODR(mydata, linear, beta0 = linreg[0:2])
        myoutput = myodr.run()
        
        self.betas = myoutput.beta
        
    def predict(self, x):
        
        return self.model(self.betas, x)
    
    def score(self, x, y):
        
        s = np.cov(x, y)
        r = s[0,1]/np.sqrt(s[0,0]*s[1,1])
        
        return r**2

def load_subsets(ps_fn, modis_fn, ps_bands, modis_bands):
    ps = gdal.Open(ps_fn)    
    modis = gdal.Open(modis_fn)
    
    bbox = intersect_raster_envelopes([ps, modis])

    ps_sub = gp.get_subset(bbox, ps, greedy = True)
    pixels = ps_sub.pixels() / 2**12 # Scale
    
    ps_sub.write_pixels(pixels[ps_bands])
    ps_sub.set_projection(ps.GetProjectionRef())
    
    modis_sub = gp.get_subset(bbox, modis, greedy = True)
    mpixels = modis_sub.pixels()
    mpixels_scaled = mpixels * .0001
    mpixels_scaled[mpixels == -28672] = 0.
    
    modis_sub.write_pixels(mpixels_scaled[modis_bands])
    modis_sub.set_projection(modis.GetProjectionRef())
    
    return ps_sub, modis_sub

def mask_nodata(images):
    # Treat zeros as nodata areas in each image
    mask = images[0].pixels()[0]
    
    for image in images[1:]:
        mask = mask * image.pixels()[0]
    
    mask[mask > 0] = 1
    mask = mask.astype(bool)
    
    masked_images = []

    for image in images:
        pixels = image.pixels()

        for b in xrange(image.get_band_count()):
            band = pixels[b]
            band[~mask] = 0.
            pixels[b] = band
        
        image.write_pixels(pixels)

    return mask

def intersect_raster_envelopes(imgs):
    
    bboxs = []
    
    for img in imgs:
        if hasattr(img, 'get_gt'): 
            bbox = gp.get_geo_extent(img.get_gt(), (0, 0) + (
                img.get_column_count(),
                img.get_row_count()))
            
        elif hasattr(img, 'GetGeoTransform'):
            bbox = gp.get_geo_extent(img.GetGeoTransform(), (0, 0) + (
                img.RasterXSize,
                img.RasterYSize))
            
        bboxs.append(bbox)
        
    geom = intersect_bboxs(bboxs)
    return geom.GetEnvelope()
        
def intersect_bboxs(bboxs):
    igeom = bbox_to_geom(bboxs[0])
    
    for bbox in bboxs[1:]:
        ngeom = bbox_to_geom(bbox)
        igeom = igeom.Intersection(ngeom)
        
    return igeom
            
def bbox_to_geom(bbox):
    xmin, xmax, ymin, ymax = bbox
    wkt = ('POLYGON (({0} {1}, {0} {3}, {2} {3}, {2} {1}, {0} {1}))'
               .format(xmin, ymin, xmax, ymax))
        
    return ogr.CreateGeometryFromWkt(wkt)
    

def align_to(master, to_align):
    ''' 
        Takes Subsets as input (gpixel specific)
    '''
    from_gt = master.get_gt()
    to_gt = to_align.get_gt()
    
    to_align_pix = to_align.pixels()
    ny = master.get_row_count()
    nx = master.get_column_count()
    nb = to_align.get_band_count()
    
    aligned = np.zeros((nb, ny, nx))
    
    for b in xrange(nb):
        aligned[b] = align_image_grids((ny, nx), to_align_pix[b], 
               from_gt, to_gt, order = 1)
        
    sub = gp.Subset(from_gt, aligned)
    #sub.set_projection()
    return sub

def align_image_grids(src_shape, to_align, gt_src, gt_dst, order = 1):
    # Mapping function for pixel coords between images
    def p2p_x(a, b, c, g, h, px, py):
        return ((a + px * b + py * c) - g) / h
      
    def p2p_y(d, e, f, j, l, px, py):
        return ((d + px * e + py * f) - j) / l

    x2x = np.vectorize(p2p_x)
    y2y = np.vectorize(p2p_y)
    
    y, x = np.indices(src_shape)

    a,b,c,d,e,f = gt_src
    g,h,i,j,k,l = gt_dst

    x_pix = x2x(a, b, c, g, h, x, y)
    y_pix = y2y(d, e, f, j, l, x, y)

    mapping = [y_pix, x_pix]
    
    return ndimage.map_coordinates(to_align, mapping, order = order)

def fit_linear_model(x, y):
    
    model = linear_model.RANSACRegressor()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    model.fit(x, y)
    
    if hasattr(model, 'inlier_mask_'):
        inliers = model.inlier_mask_
        x = x[inliers]
        y = y[inliers]

    y_pred = model.predict(x)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mae_r = mae / y.mean()
    
    return model, r2, mae, mae_r

def scale(rgb, n = 2., method = 'sd', ignore_zero = False):
    
    warnings.filterwarnings('ignore')
        
    scaled = rgb.copy()
    
    for b in xrange(len(rgb)):
        
        band = scaled[b]
        
        if ignore_zero:
            band = band[band > 0.]
            
        sd = np.std(band) * n
        mn = np.mean(band)
        dx = sd * 2
        scaled[b] = (rgb[b] - (mn - sd)) * (1 / dx)
        
    scaled[scaled < 0.] = 0.
    scaled[scaled > 1.] = 1.
    
    if ignore_zero:
            scaled[rgb == 0.] = 0.
    
    return scaled

def downsample_to_raster(src, dst, agg_func = np.mean, align_grids = True):
    
    src_res = np.array(src.get_resolution())
    dst_res = np.array(dst.get_resolution())
    
    sfactor = np.ceil(dst_res / src_res).astype(int)
    
    agg = aggregate(src, sfactor, agg_func)
    
    if align_grids:
        agg = align_to(dst, agg)
        
    return agg
    
def aggregate(subset, factor, agg_func = np.mean):
    agg_gt = np.array(subset.get_gt())
    xres, yres = subset.get_resolution()
    
    xf, yf = factor
    agg_gt[1] = xf * xres
    agg_gt[5] = yf * yres
    
    agg = block_reduce(subset.pixels(), block_size = (1, yf, xf), func = agg_func)
    
    return gp.Subset(tuple(agg_gt), agg)

def imad(img1, img2):
       
    # Dimensions, assume they match!
    bands, rows, cols = img1.shape
    
    # Matched dimensions so;
    dims = [0,0,cols,rows]
    x10,y10,cols1,rows1 = dims
    x20,y20,cols,rows = dims
    
    lam = 0.0
    
    cpm = Cpm(2*bands)    
    delta = 1.0
    oldrho = np.zeros(bands)     
    itr = 0
    tile = np.zeros((cols,2*bands))
    sigMADs = 0
    means1 = 0
    means2 = 0
    A = 0
    B = 0
    rasterBands1 = img1
    rasterBands2 = img2
                   
    while (delta > 0.001) and (itr < 100):   
#      spectral tiling for statistics
        for row in range(rows):
            for k in range(bands):
                tile[:,k] = rasterBands1[k][y10+row, x10:cols]
                tile[:,bands+k] = rasterBands2[k][y10+row, x10:cols]
#          eliminate no-data pixels (assuming all zeroes)                  
            tst1 = np.sum(tile[:,0:bands],axis=1) 
            tst2 = np.sum(tile[:,bands::],axis=1) 
            idx1 = set(np.where((tst1>0))[0]) 
            idx2 = set(np.where((tst2>0))[0]) 
            idx = list(idx1.intersection(idx2))    
            if itr>0:
                mads = np.asarray((
                    tile[:,0:bands]-means1)*A - (tile[:,bands::]-means2)*B
                )
                chisqr = np.sum((mads/sigMADs)**2,axis=1)
                wts = 1-stats.chi2.cdf(chisqr,[bands])
                cpm.update(tile[idx,:],wts[idx])
            else:
                cpm.update(tile[idx,:])               

#     weighted covariance matrices and means 
        S = cpm.covariance() 
        means = cpm.means()    
#     reset prov means object           
        cpm.__init__(2*bands)  
        s11 = S[0:bands,0:bands]
        s11 = (1-lam)*s11 + lam*np.eye(bands)
        s22 = S[bands:,bands:] 
        s22 = (1-lam)*s22 + lam*np.eye(bands)
        s12 = S[0:bands,bands:]
        s21 = S[bands:,0:bands]        
        c1 = s12*linalg.inv(s22)*s21 
        b1 = s11
        c2 = s21*linalg.inv(s11)*s12
        b2 = s22
#     solution of generalized eigenproblems 
        if bands>1:
            mu2a,A = geneiv(c1,b1)                
            mu2b,B = geneiv(c2,b2)               
#          sort a   
            idx = np.argsort(mu2a)
            A = A[:,idx]        
#          sort b   
            idx = np.argsort(mu2b)
            B = B[:,idx] 
            mu2 = mu2b[idx]
        else:
            mu2 = c1/b1
            A = 1/np.sqrt(b1)
            B = 1/np.sqrt(b2)   
#      canonical correlations             
        mu = np.sqrt(mu2)
        a2 = np.diag(A.T*A)
        b2 = np.diag(B.T*B)
        sigma = np.sqrt( (2-lam*(a2+b2))/(1-lam)-2*mu )
        rho=mu*(1-lam)/np.sqrt( (1-lam*a2)*(1-lam*b2) )
#      stopping criterion
        delta = max(abs(rho-oldrho))
        #print delta,rho 
        oldrho = rho  
#      tile the sigmas and means             
        sigMADs = np.tile(sigma,(cols,1)) 
        means1 = np.tile(means[0:bands],(cols,1)) 
        means2 = np.tile(means[bands::],(cols,1))
#      ensure sum of positive correlations between X and U is positive
        D = np.diag(1/np.sqrt(np.diag(s11)))  
        s = np.ravel(np.sum(D*s11*A,axis=0)) 
        A = A*np.diag(s/np.abs(s))          
#      ensure positive correlation between each pair of canonical variates        
        cov = np.diag(A.T*s12*B)    
        B = B*np.diag(cov/np.abs(cov))          
        itr += 1
    
    out_bands = np.zeros((bands+1, rows, cols)) 
    
    for row in range(rows):
        for k in range(bands):
            tile[:,k] = rasterBands1[k][y10+row, x10:cols]
            tile[:,bands+k] = rasterBands2[k][y10+row, x10:cols]     
        mads = np.asarray((tile[:,0:bands]-means1)*A - (tile[:,bands::]-means2)*B)
        chisqr = np.sum((mads/sigMADs)**2,axis=1) 
        out_bands[bands, y10+row,  x10:cols] = np.reshape(chisqr,(1,cols))  
        
        for k in range(bands):
            out_bands[k, y10+row,  x10:cols] = np.reshape(mads[:,k],(1,cols))
            
    return out_bands
    
class Cpm(object):
    '''Provisional means algorithm'''
    def __init__(self,N):
        self.mn = np.zeros(N)
        self.cov = np.zeros((N,N))
        self.sw = 0.0000001

    def update(self,Xs,Ws=None):
        n,N = np.shape(Xs)
        if Ws is None:
            Ws = np.ones(n)
        sw = ctypes.c_double(self.sw)
        mn = self.mn
        cov = self.cov
        provmeans(Xs,Ws,N,n,ctypes.byref(sw),mn,cov)
        self.sw = sw.value
        self.mn = mn
        self.cov = cov

    def covariance(self):
        c = np.mat(self.cov/(self.sw-1.0))
        d = np.diag(np.diag(c))
        return c + c.T - d

    def means(self):
        return self.mn

def choldc(A):
# Cholesky-Banachiewicz algorithm,
# A is a numpy matrix
    L = A - A
    for i in range(len(L)):
        for j in range(i):
            sm = 0.0
            for k in range(j):
                sm += L[i,k]*L[j,k]
            L[i,j] = (A[i,j]-sm)/L[j,j]
        sm = 0.0
        for k in range(i):
            sm += L[i,k]*L[i,k]
        L[i,i] = math.sqrt(A[i,i]-sm)
    return L

def geneiv(A,B):
# solves A*x = lambda*B*x for numpy matrices A and B,
# returns eigenvectors in columns
    Li = np.linalg.inv(choldc(B))
    C = Li*A*(Li.transpose())
    C = np.asmatrix((C + C.transpose())*0.5,np.float32)
    eivs,V = np.linalg.eig(C)
    return eivs, Li.transpose()*V