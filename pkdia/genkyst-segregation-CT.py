from exams.exam_genkyst import exam_genkyst
import numpy as np
from openpyxl import load_workbook
import random
from current import current_xlsx
from sklearn.model_selection import StratifiedKFold, train_test_split

def add_longitudinal_data(split_ids, list_ids, list_series):
    new_list_ids, new_list_series = list(), list()
    for id_ in split_ids:
        for pos in np.where(list_ids==id_)[0]:
            exam = exam_genkyst(list_ids[pos], int(list_series[pos]), None, False)
            if exam.CT_exist and exam.kid_annot:
                new_list_ids.append(list_ids[pos])
                new_list_series.append(list_series[pos])
    return np.array(new_list_ids), np.array(new_list_series)
        
wb = load_workbook(current_xlsx()[0])
sh = wb['main']
list_ids = np.array([sh.cell(row=rownum, column=1).value for rownum in range(2,sh.max_row+1)], dtype=int)
list_series = np.array([sh.cell(row=rownum, column=8).value for rownum in range(2,sh.max_row+1)], dtype=int)

ids, ser, mayo = list(), list(), list()

for idx, id_ in enumerate(list_ids):
    exam = exam_genkyst(int(id_), list_series[idx], None, False)
    if exam.CT_exist and exam.kid_annot:
        if id_ not in ids:
            ids.append(id_)
            if exam.mayo == '2A' or exam.mayo == '2B':
                mayo.append('2')
            else:
                mayo.append(exam.mayo)
ids, mayo = np.array(ids), np.array(mayo)
print(np.unique(mayo))

print('pourcentage of 1A: %d'%(len(np.where(mayo=='1A')[0])))
print('pourcentage of 1B: %d'%(len(np.where(mayo=='1B')[0])))
print('pourcentage of 1C: %d'%(len(np.where(mayo=='1C')[0])))
print('pourcentage of 1D: %d'%(len(np.where(mayo=='1D')[0])))
print('pourcentage of 1E: %d'%(len(np.where(mayo=='1E')[0])))
print('pourcentage of 2: %d'%(len(np.where(mayo=='2')[0])))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(ids, mayo), start=1):
    
    train_ids, test_ids = ids[train_index], ids[test_index]

    train_mayo, test_mayo = mayo[train_index], mayo[test_index]

    train_ids, val_ids, train_mayo, val_mayo = train_test_split(train_ids, train_mayo, test_size=0.1, 
                                                                stratify=train_mayo, random_state=fold)

    train_ids, train_series = add_longitudinal_data(train_ids, list_ids, list_series)
    val_ids, val_series     = add_longitudinal_data(val_ids, list_ids, list_series)
    test_ids, test_series   = add_longitudinal_data(test_ids, list_ids, list_series)
    
    print('fold ', fold)
    print(len(train_ids), len(val_ids), len(test_ids))
    print(train_ids.tolist())
    print(train_series.tolist())
    print(val_ids.tolist())
    print(val_series.tolist())
    print(test_ids.tolist())
    print(test_series.tolist())