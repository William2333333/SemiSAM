import xlrd

# == identify CT scans from berlin with LV ground truth annotations
root = './../../data/berlin/'
xlsx = root+'2024-05-05-berlin-dataset.xlsx'

wb = xlrd.open_workbook(xlsx)
sh = wb.sheet_by_name('main')
list_ids = [list(sh.row_values(rownum))[0] for rownum in range(1,sh.nrows)]
list_series = [int(list(sh.row_values(rownum))[5]) for rownum in range(1,sh.nrows)]
list_CT = [list(sh.row_values(rownum))[8] for rownum in range(1,sh.nrows)]

b_ids_CT, b_series_CT = list(), list()
for idx, id_ in enumerate(list_ids): 
    if list_CT[idx] == 'x':
        b_ids_CT.append(id_)
        b_series_CT.append(list_series[idx])
print(b_ids_CT)
# ['007', '007', '017', '029', '029', '030', '035', '041', '043', '044']
print(b_series_CT)
# [1, 3, 1, 1, 2, 1, 1, 1, 1, 1]
# ==

# == identify CT scans from genkyst with LV ground truth annotations
root = './../../data/genkyst/'
xlsx = root+'2024-01-10-genkyst-dataset.xlsx'

wb = xlrd.open_workbook(xlsx)
sh = wb.sheet_by_name('main')
list_ids = [list(sh.row_values(rownum))[0] for rownum in range(1,sh.nrows)]
list_series = [int(list(sh.row_values(rownum))[7]) for rownum in range(1,sh.nrows)]
list_CT = [list(sh.row_values(rownum))[11] for rownum in range(1,sh.nrows)]
list_liver = [list(sh.row_values(rownum))[17] for rownum in range(1,sh.nrows)]

g_ids_CT, g_series_CT = list(), list()
for idx, id_ in enumerate(list_ids): 
    if list_CT[idx] == 'x' and list_liver[idx] == 'x':
        g_ids_CT.append(id_)
        g_series_CT.append(list_series[idx])
print(g_ids_CT)
# ['010071', '010171', '019289', '019591', '019591', '031283', '031283', '038266', '048439', '068996', '208839', '300003', '300004', '300005']
print(g_series_CT)
# [1, 3, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1]
# ==
