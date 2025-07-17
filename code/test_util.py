import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm

from skimage.measure import label

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        if id[10:12] == "LK":
            kidney = "LK"
        else:
            kidney = "RK"
        id = id[:9]

        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        prediction = getLargestCC(prediction)  #########

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            print(f"{id}_{kidney} metric: " , single_metric)
        total_metric += np.asarray(single_metric)

        if save_result:

            t2_img = nib.load(f'/home/liao/data/genkyst/dataset/{id}-T2.nii.gz')  # 整个人体MRI
            kidney_cyst_img = nib.load(f"/home/liao/code/aimarcs-SAM/SAM/test_SAM/3D_mask/{id}_{kidney}_3D_csyt.nii.gz")
            kidney_img = nib.load(f'/home/liao/data/genkyst/dataset/{id}-{kidney}.nii.gz')  # 左肾掩码
            t2_data = t2_img.get_fdata()
            kidney_data = kidney_img.get_fdata()
            kidney_cyst_data = kidney_cyst_img.get_fdata()
            indices = np.where(kidney_data != 0)
            if len(indices[0]) == 0:
                raise ValueError("掩码中没有检测到非零数据，请检查左肾掩码的内容。")
            min_x, max_x = np.min(indices[0]), np.max(indices[0])
            min_y, max_y = np.min(indices[1]), np.max(indices[1])
            min_z, max_z = np.min(indices[2]), np.max(indices[2])
            cropped_t2_data = t2_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
            cropped_kidney_cyst_label_data = kidney_cyst_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
            orig_affine = t2_img.affine
            voxel_offset = np.array([min_x, min_y, min_z, 1])
            new_origin = orig_affine @ voxel_offset  # 矩阵乘法，得到新的原点位置
            new_affine = orig_affine.copy()
            new_affine[:3, 3] = new_origin[:3]

            nib.save(nib.Nifti1Image(prediction.astype(np.float32), new_affine), test_save_path + id + "_" + kidney +"_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), new_affine), test_save_path + id + "_" + kidney + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), new_affine), test_save_path + id + "_" + kidney +"_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
