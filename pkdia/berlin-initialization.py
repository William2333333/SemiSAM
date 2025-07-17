#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from exams.exam_berlin import exam_berlin
import xlrd
import nibabel
from utils.utils import mask_init_array
import numpy as np

root = './../../data/berlin/'
xlsx = root+'2024-05-05-berlin-dataset.xlsx'

wb = xlrd.open_workbook(xlsx)
sh = wb.sheet_by_name('main')
list_ids = [list(sh.row_values(rownum))[0] for rownum in range(1,sh.nrows)]
list_series = [int(list(sh.row_values(rownum))[5]) for rownum in range(1,sh.nrows)]
list_T2c = [list(sh.row_values(rownum))[6] for rownum in range(1,sh.nrows)]
list_T2a = [list(sh.row_values(rownum))[7] for rownum in range(1,sh.nrows)]
list_CT = [list(sh.row_values(rownum))[8] for rownum in range(1,sh.nrows)]
print(list_ids, list_series, list_T2c, list_T2a, list_CT)

for idx, id_ in enumerate(list_ids):
    if list_T2c[idx] == 'x':
        exam = exam_berlin(int(id_), list_series[idx], 'T2c')
        nibabel.save(exam.T2c,exam.folder+exam.id+'-%02d.nii.gz'%exam.serie)
    elif list_T2a[idx] == 'x':
        exam = exam_berlin(int(id_), list_series[idx], 'T2a')
        nibabel.save(exam.T2a,exam.folder+exam.id+'-%02d.nii.gz'%exam.serie)
    elif list_CT[idx] == 'x':
        exam = exam_berlin(int(id_), list_series[idx], 'CT')
        nibabel.save(exam.CT,exam.folder+exam.id+'-%02d.nii.gz'%exam.serie)
    array = exam.LV.get_fdata()>0
    array = np.array(array, dtype=int)
    print(np.unique(array))
    if list_T2c[idx] == 'x':
        nibabel.save(mask_init_array(exam.T2c, array), exam.folder+exam.id+'-%02d-LV.nii.gz'%exam.serie)
    elif list_T2a[idx] == 'x':
        mask_init_array(exam.T2a, array)
        nibabel.save(mask_init_array(exam.T2a, array), exam.folder+exam.id+'-%02d-LV.nii.gz'%exam.serie)
    elif list_CT[idx] == 'x':
        mask_init_array(exam.CT, array)
        nibabel.save(mask_init_array(exam.CT, array), exam.folder+exam.id+'-%02d-LV.nii.gz'%exam.serie)