#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from exams.exam_genkyst import exam_genkyst
import distutils.dir_util
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.transform import resize, rotate
from skimage import io, img_as_ubyte
import tqdm
from utils import utils

import os
import nibabel as nib
import matplotlib.pyplot as plt





    


def extract_genkyst_slice_MT(exam, xyz, modality, size):
    if modality == 'T2':
        img = rotate(resize(np.squeeze(exam.T2.get_fdata()[:,xyz,:])[::-1,:], output_shape=size, preserve_range=True), 90, preserve_range=True)
        mask_LK = rotate(resize(exam.LK.get_fdata()[:,xyz,:][::-1,:]*255, output_shape=size, preserve_range=True), 90, preserve_range=True)
        mask_RK = rotate(resize(exam.RK.get_fdata()[:,xyz,:][::-1,:]*255, output_shape=size, preserve_range=True), 90, preserve_range=True)
    elif modality == 'CT':
        img = rotate(resize(np.squeeze(exam.CT.get_fdata()[:,:,xyz])[::-1,:], output_shape=size, preserve_range=True), 90, preserve_range=True)
        mask_LK = rotate(resize(exam.LK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=size, preserve_range=True), 90, preserve_range=True)
        mask_RK = rotate(resize(exam.RK.get_fdata()[:,:,xyz][::-1,:]*255, output_shape=size, preserve_range=True), 90, preserve_range=True)
    min_greyscale, max_greyscale = np.percentile(img,(1,99)) 
    img = rescale_intensity(img, in_range=(min_greyscale,max_greyscale), out_range=(0,1))
    mask_LK[np.where(mask_LK>0)] = 255
    mask_RK[np.where(mask_RK>0)] = 255
    return img_as_ubyte(img), mask_LK.astype(np.uint8), mask_RK.astype(np.uint8)

class Exam:
    pass


patience_id = "010005-01"
t2_path = f"/home/liao/data/genkyst/dataset/{patience_id}-T2.nii.gz"
lk_path = f"/home/liao/data/genkyst/dataset/{patience_id}-LK.nii.gz"
rk_path = f"/home/liao/data/genkyst/dataset/{patience_id}-RK.nii.gz"

exam = Exam()
exam.T2 = nib.load(t2_path)
exam.LK = nib.load(lk_path)
exam.RK = nib.load(rk_path)

t2_data = exam.T2.get_fdata()
num_slices = t2_data.shape[1]
print(f"total number of slices is {num_slices}")

target_size = (492,642)

output_folder_LK = f"/home/liao/code/aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized/exam_{patience_id}/LK"
os.makedirs(output_folder_LK,exist_ok=True)
output_folder_RK = f"/home/liao/code/aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized/exam_{patience_id}/RK"
os.makedirs(output_folder_RK,exist_ok=True)


for slice_index in range(num_slices):
    _,mask_LK, mask_RK = extract_genkyst_slice_MT(exam,slice_index,'T2',target_size)

    io.imsave(os.path.join(output_folder_LK,f"slice_{slice_index:03d}_LK.png"),mask_LK)
    io.imsave(os.path.join(output_folder_RK,f"slice_{slice_index:03d}_RK.png"),mask_RK)

print("all results saved")


