#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel
import numpy as np
from utils.utils import mask_zero

# == cropping too large images == #
"""
from utils.utils import img_init_array

path   = './../../data/genkyst/test/' #
id_    = 42293
serie  = 1
modal  = 'CT'

src = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz'))

new_src = img_init_array(src, src.dataobj[:,:,200:850])
    
nibabel.save(nibabel.as_closest_canonical(new_src), path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz')

LK = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LK.nii.gz'))

new_LK = img_init_array(new_src, LK.dataobj[:,:,200:850])
    
nibabel.save(nibabel.as_closest_canonical(new_LK), path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LK.nii.gz')

RK = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'RK.nii.gz'))

new_RK = img_init_array(new_src, RK.dataobj[:,:,200:850])
    
nibabel.save(nibabel.as_closest_canonical(new_RK), path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'RK.nii.gz')
"""
# == case where we have only the liver mask which is provided == #

"""
path   = './../../../data/genkyst/new/'
id_    = 300003
serie  = 1
modal  = 'CT'

src = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz'))
    
nibabel.save(src, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz')

mask_LV = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LV.nii.gz'))

new_mask = mask_zero(mask_LV)

new_mask.dataobj[np.where(mask_LV.get_fdata()>0)] = 1
  
nibabel.save(new_mask, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LV.nii.gz')
"""

# == case where we have RK, LK and LV mask provided == #

"""
path   = './../../../data/genkyst/new/'
id_    = 100015
serie  = 1
modal  = 'T2'
LK_idx = 5
RK_idx = 2
LV_idx = 1

src = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz'))
    
nibabel.save(src, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz')

mask_ALL = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'ALL.nii.gz'))

mask_LK, mask_RK, mask_LV = mask_zero(mask_ALL), mask_zero(mask_ALL), mask_zero(mask_ALL)

mask_LK.dataobj[np.where(mask_ALL.get_fdata()==float(LK_idx))] = 1

mask_RK.dataobj[np.where(mask_ALL.get_fdata()==float(RK_idx))] = 1

mask_LV.dataobj[np.where(mask_ALL.get_fdata()==float(LV_idx))] = 1

nibabel.save(mask_LK, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LK.nii.gz')

nibabel.save(mask_RK, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'RK.nii.gz')

nibabel.save(mask_LV, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LV.nii.gz')
"""

# == case where we have RK and LK provided (in a single BK file) == #
"""
path   = './../../../data/genkyst/new/'
id_    = 10076
serie  = 2
modal  = 'T2'
LK_idx = 1
RK_idx = 1

src = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz'))
    
nibabel.save(src, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz')

mask_BK = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'BK.nii.gz'))

mask_LK, mask_RK = mask_zero(mask_BK), mask_zero(mask_BK)

mask_LK.dataobj[np.where(mask_BK.get_fdata()==float(LK_idx))] = 1

mask_RK.dataobj[np.where(mask_BK.get_fdata()==float(RK_idx))] = 1

nibabel.save(mask_LK, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LK.nii.gz')

nibabel.save(mask_RK, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'RK.nii.gz')
"""

# == case where we have RK and LK provided (in separate files) == #

path   = './../../data/genkyst/dataset/'
id_    = 108192
serie  = 1
modal  = 'T2'

src = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz'))
    
nibabel.save(src, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+modal+'.nii.gz')

mask_LK = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LK.nii.gz'))

mask_RK = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'RK.nii.gz'))

mask_LV = nibabel.as_closest_canonical(nibabel.load(path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LV.nii.gz'))
mask_LV_ = mask_zero(mask_LK)
LV_idx = 4
mask_LV_.dataobj[np.where(mask_LV.get_fdata()==float(LV_idx))] = 1
nibabel.save(mask_LV_, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LV.nii.gz')

nibabel.save(mask_LK, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'LK.nii.gz')

nibabel.save(mask_RK, path+'%0*d'%(6,id_)+'-%0*d-'%(2,serie)+'RK.nii.gz')
