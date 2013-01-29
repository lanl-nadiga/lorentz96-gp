import os, sys, itertools
import numpy as np
from multiprocessing import *
import gpsm
import pymc


def improv_p2h():
    nprms = gpsm.model.nprms
    npts = len(gpsm.mesh_high.indices)
    p2h_in = gpsm.mesh_low.d1[gpsm.mesh_high.indices,:nprms] #advanced indexing. Therefore a copy
    p2h_tr_full = np.hstack((p2h_in,gpsm.mesh_high.pca_output[:,0].reshape(npts,1)))

    other_args = [p2h_tr_full]

    diff_deg_val = 1.0
    ddShared = Array['d', diff_deg_val, lock=False]
    pairs = [[i,j] for i in range(npts) for j in range(npts) if i!=j] #HAS to be list of LISTS, NOT TUP

    ncpu = cpu_count()
    ncpu = 1
    p = Pool(ncpu)
    foo = p.map(gpsm.like_pair_1map,zip(pairs,itertools.repeat(other_args,len(pairs))))
    
    return np.array(foo)

	


def improv_p2l2h_lo():
    nprms = gpsm.model.nprms
    npts_p2l = len(gpsm.mesh_low.indices)
    p2l_in = gpsm.mesh_low.d1[:,:nprms] 
    p2l_tr_full = np.hstack((p2l_in,gpsm.mesh_low.pca_output[:,0].reshape(npts_p2l,1)))

    npts_l2h = len(gpsm.mesh_high.indices)
    l2h_in = gpsm.mesh_low.d1[gpsm.mesh_high.indices,:nprms+1] #Also has prms
    l2h_tr_full = np.hstack((l2h_in,gpsm.mesh_high.pca_output[:,0].reshape(npts_l2h,1)))

    other_args = [p2l_tr_full, l2h_tr_full]
    
    pairs = [[i,j] for i in range(npts_p2l) for j in range(npts_l2h)  #HAS to be list of LISTS, NOT TUP
             if i != gpsm.mesh_high.indices[j]] 

    p = Pool(12)
    foo = p.map(gpsm.unlike_pair,zip(pairs,itertools.repeat(other_args,len(pairs))))
    
    return np.array(foo)


def improv_p2l2h_hi():
    nprms = gpsm.model.nprms
    npts_p2l = len(gpsm.mesh_low.indices)
    p2l_in = gpsm.mesh_low.d1[:,:nprms] 
    p2l_tr_full = np.hstack((p2l_in,gpsm.mesh_low.pca_output[:,0].reshape(npts_p2l,1)))

    npts_l2h = len(gpsm.mesh_high.indices)
    l2h_in = gpsm.mesh_low.d1[gpsm.mesh_high.indices,:nprms+1] #Also has prms
    l2h_tr_full = np.hstack((l2h_in,gpsm.mesh_high.pca_output[:,0].reshape(npts_l2h,1)))

    other_args = [p2l_tr_full, l2h_tr_full]
    
    pairs = [[i,j] for i in range(npts_l2h) for j in range(npts_l2h)  #HAS to be list of LISTS, NOT TUP
             if i != j] 

    p = Pool(12)
    foo = p.map(gpsm.like_pair,zip(pairs,itertools.repeat(other_args,len(pairs))))
    
    return np.array(foo)


# foo1 = improv_p2h().flatten()
# foo2 = improv_p2l2h_lo().flatten()
foo3 = improv_p2l2h_hi().flatten()

foo = foo3

foo = foo[abs(foo-mean(foo))<2*std(foo)]
figure()
hist(foo,50)
title('%.2f'%foo.mean())
pass
	
