#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel
import logging
from utils.utils import normalization_imgs
import os

class exam_genkyst_prod: # PKDIAv2

    def __init__(self, id_, serie, path, output, modality):
        
        self.id = '%0*d'%(6,id_)
        self.serie = serie
        self.path = path
        self.output = output
        self.modality = modality
        self.exam_upload()
        self.print_info()

    def exam_upload(self):

        if self.modality == 'T2':
            self.T2 = nibabel.as_closest_canonical(nibabel.load(self.path+self.id+'-%02d-T2.nii.gz'%self.serie))
            if os.path.exists(self.output+self.id+'-%02d-T2-prod.nii.gz'%self.serie) == False:
                nibabel.save(self.T2, self.output+self.id+'-%02d-T2-prod.nii.gz'%self.serie)
        elif self.modality == 'CT':
            self.CT = nibabel.as_closest_canonical(nibabel.load(self.path+self.id+'-%02d-CT.nii.gz'%self.serie))
            if os.path.exists(self.output+self.id+'-%02d-CT-prod.nii.gz'%self.serie) == False:
                nibabel.save(self.CT, self.output+self.id+'-%02d-CT-prod.nii.gz'%self.serie)            
        
    def normalize(self):
        if self.modality == 'T2':
            self.T2.get_fdata()[:,:,:] = normalization_imgs(self.T2.get_fdata())[:,:,:]
        elif self.modality == 'CT':
            self.CT.get_fdata()[:,:,:] = normalization_imgs(self.CT.get_fdata())[:,:,:]

    def print_info(self):
        
        logging.basicConfig(level=logging.INFO, format='\n %(levelname)s: %(message)s')
        logging.info(f'''exam {self.id} uploaded:
        serie:         {self.serie}
        ''')