#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
from skimage import io
import numpy as np
import os
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, affine
from exams.exam_genkyst import exam_genkyst
from exams.exam_genkyst_prod import exam_genkyst_prod
from manage.manage_genkyst import extract_genkyst_slice, extract_genkyst_slice_prod
from utils.utils import normalization_imgs, normalization_masks
 
class dataset_genkyst(Dataset):

    def __init__(self, path:str, scheme:str, modality:str, anatomy:str, vgg:bool=False):
        self.path = path
        self.scheme = scheme
        self.modality = modality 
        self.anatomy = anatomy
        self.vgg = vgg
        self.ids = np.load(self.path+'imgs-id-'+self.scheme+'.npy')

    def __len__(self):
        return len(self.ids)

    def transform(self, img, mask):
        (d,t,sc,sh) = transforms.RandomAffine.get_params(degrees=(-20,20), translate=(0.2,0.2), scale_ranges=(0.8,1.2), shears=(-20,20), img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        mask = affine(to_pil_image(mask), angle=d, translate=t, scale=sc, shear=sh)
        return (np.array(img), np.array(mask))

    def __getitem__(self, idx:int):
        folder = self.path+self.scheme+'/'
        img = io.imread(folder+self.ids[idx]+'-src.png')
        mask = io.imread(folder+self.ids[idx]+'-mask.png')
        if self.scheme == 'train':
            img, mask = self.transform(img, mask)	
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_ = np.zeros(shape=(mask.shape[0],mask.shape[1],1), dtype=np.uint8)
        mask_[:,:,0] = mask
        img_ = normalization_imgs(img_)
        if self.anatomy != 'BK2C': # no normalization of segmentation mask when 2 classes
            mask_ = normalization_masks(mask_)
        return (img_.swapaxes(2,0), mask_.swapaxes(2,0))
    
class dataset_genkyst_MT(Dataset):

    def __init__(self, path:str, scheme:str, modality:str, vgg:bool=False):
        self.path = path
        self.scheme = scheme
        self.modality = modality 
        self.anatomy = 'LK+RK'
        self.vgg = vgg
        self.ids = np.load(self.path+'imgs-id-'+self.scheme+'.npy')

    def __len__(self):
        return len(self.ids)

    def transform(self, img, mask_LK, mask_RK):
        (d,t,sc,sh) = transforms.RandomAffine.get_params(degrees=(-20,20), translate=(0.2,0.2), scale_ranges=(0.8,1.2), shears=(-20,20), img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        mask_LK = affine(to_pil_image(mask_LK), angle=d, translate=t, scale=sc, shear=sh)
        mask_RK = affine(to_pil_image(mask_RK), angle=d, translate=t, scale=sc, shear=sh)
        return (np.array(img), np.array(mask_LK), np.array(mask_RK))

    def __getitem__(self, idx:int):
        folder = self.path+self.scheme+'/'
        img = io.imread(folder+self.ids[idx]+'-src.png')
        mask_LK = io.imread(folder+self.ids[idx]+'-mask-LK.png')
        mask_RK = io.imread(folder+self.ids[idx]+'-mask-RK.png')
        if self.scheme == 'train':
            img, mask_LK, mask_RK = self.transform(img, mask_LK, mask_RK)	
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_LK_ = np.zeros(shape=(mask_LK.shape[0],mask_LK.shape[1],1), dtype=np.uint8)
        mask_LK_[:,:,0] = mask_LK
        mask_RK_ = np.zeros(shape=(mask_RK.shape[0],mask_RK.shape[1],1), dtype=np.uint8)
        mask_RK_[:,:,0] = mask_RK
        img_ = normalization_imgs(img_)
        mask_LK_ = normalization_masks(mask_LK_)
        mask_RK_ = normalization_masks(mask_RK_)
        return (img_.swapaxes(2,0), mask_LK_.swapaxes(2,0), mask_RK_.swapaxes(2,0))
    
class dataset_genkyst_3O(Dataset):

    def __init__(self, path:str, scheme:str, modality:str, vgg:bool=False):
        self.path = path
        self.scheme = scheme
        self.modality = modality 
        self.anatomy = '30'
        self.vgg = vgg
        self.ids = np.load(self.path+'imgs-id-'+self.scheme+'.npy')

    def __len__(self):
        return len(self.ids)

    def transform(self, img, mask_LK, mask_RK, mask_LV):
        (d,t,sc,sh) = transforms.RandomAffine.get_params(degrees=(-20,20), translate=(0.2,0.2), scale_ranges=(0.8,1.2), shears=(-20,20), img_size=img.shape)
        img = affine(to_pil_image(img), angle=d, translate=t, scale=sc, shear=sh)
        if mask_LK is not None:
            mask_LK = np.array(affine(to_pil_image(mask_LK), angle=d, translate=t, scale=sc, shear=sh))
            mask_RK = np.array(affine(to_pil_image(mask_RK), angle=d, translate=t, scale=sc, shear=sh))
        if mask_LV is not None:
            mask_LV = np.array(affine(to_pil_image(mask_LV), angle=d, translate=t, scale=sc, shear=sh))
        return (np.array(img), mask_LK, mask_RK, mask_LV)

    def __getitem__(self, idx:int):
        folder = self.path+self.scheme+'/'
        img = io.imread(folder+self.ids[idx]+'-src.png')
        mask_LK = io.imread(folder+self.ids[idx]+'-mask-LK.png')
        mask_RK = io.imread(folder+self.ids[idx]+'-mask-RK.png')
        if os.path.exists(folder+self.ids[idx]+'-mask-LV.png'): 
            mask_LV = io.imread(folder+self.ids[idx]+'-mask-LV.png')
        else:
            mask_LV = None
        if self.scheme == 'train':
            img, mask_LK, mask_RK, mask_LV = self.transform(img, mask_LK, mask_RK, mask_LV)	
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_LK_ = np.zeros(shape=(mask_LK.shape[0],mask_LK.shape[1],1), dtype=np.uint8)
        mask_LK_[:,:,0] = mask_LK
        mask_LK_ = normalization_masks(mask_LK_).swapaxes(2,0)
        mask_RK_ = np.zeros(shape=(mask_RK.shape[0],mask_RK.shape[1],1), dtype=np.uint8)
        mask_RK_[:,:,0] = mask_RK
        mask_RK_ = normalization_masks(mask_RK_).swapaxes(2,0)
        if mask_LV is not None:
            mask_LV_ = np.zeros(shape=(mask_LV.shape[0],mask_LV.shape[1],1), dtype=np.uint8)
            mask_LV_[:,:,0] = mask_LV
            mask_LV_ = normalization_masks(mask_LV_).swapaxes(2,0)
        img_ = normalization_imgs(img_).swapaxes(2,0)
        if mask_LV is None:
            return (img_, mask_LK_, mask_RK_)
        else:
            return (img_, mask_LK_, mask_RK_, mask_LV_)
    
class tiny_dataset_genkyst(Dataset):
    ''' one single examination for prediction purposes '''
    
    def __init__(self, id_, serie, size, modality, anatomy:str, vgg:bool=False):
        self.id = id_
        self.serie = serie
        self.size = size
        self.modality = modality 
        self.anatomy = anatomy
        self.vgg = vgg
        self.exam = exam_genkyst(self.id, self.serie, self.modality, upload=True)
        self.exam.normalize(self.modality)
            
    def __len__(self):
        if self.modality == 'T2':
            return self.exam.T2.shape[1]
        elif self.modality == 'CT':
            return self.exam.CT.shape[2]

    def __getitem__(self, idx:int):
        img, mask = extract_genkyst_slice(self.exam, idx, self.modality, self.size, self.anatomy)
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        mask_ = np.zeros(shape=(mask.shape[0],mask.shape[1],1), dtype=np.uint8)
        mask_[:,:,0] = mask
        img_ = normalization_imgs(img_)
        mask_ = normalization_masks(mask_)
        return (img_.swapaxes(2,0), mask_.swapaxes(2,0))
    
class tiny_dataset_genkyst_prod(Dataset):
    
    def __init__(self, id_, serie, size, path, output, modality, vgg:bool=False):
        self.id = id_
        self.serie = serie
        self.size = size
        self.vgg = vgg
        self.path = path
        self.modality = modality
        self.output = output
        self.exam = exam_genkyst_prod(self.id, self.serie, self.path, self.output, self.modality)
        self.exam.normalize()
            
    def __len__(self):
        if self.modality == 'T2':
            r = self.exam.T2.shape[1]
        elif self.modality == 'CT':
            r = self.exam.CT.shape[2]
        return r

    def __getitem__(self, idx:int):
        img = extract_genkyst_slice_prod(self.exam, idx, self.size, self.modality)
        if self.vgg:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],3), dtype=np.float32)
        else:
            img_ = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.float32)
        img_[:,:,0] = img
        if self.vgg:
            img_[:,:,1], img_[:,:,2] = img, img
        img_ = normalization_imgs(img_)
        return img_.swapaxes(2,0)