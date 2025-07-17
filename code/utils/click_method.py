import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_next_click3D_torch(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):#, desc="generate points":

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        point = None
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # if no mask is given, random click a negative point
        if point is None: 
            point = torch.Tensor([np.random.randint(sz) for sz in fn_masks[i].size()]).to(torch.int64)
            is_positive = False
        bp = point[1:].clone().detach().reshape(1,1,-1).to(pred_masks.device) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1).to(pred_masks.device) 

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    


import edt
def get_next_click3D_torch_ritm(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    fn_mask_single = F.pad(fn_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fp_mask_single = F.pad(fp_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    is_positive = fn_max_dist > fp_max_dist # the biggest area is selected to be interaction point
    dt = fn_mask_dt if is_positive else fp_mask_dt
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0) # use a erosion area
    to_point_mask = to_point_mask[None, None]
    # import pdb; pdb.set_trace()


    for i in range(gt_semantic_seg.shape[0]):
        # points = torch.argwhere(to_point_mask[i])
        # points = torch.nonzero(to_point_mask[i], as_tuple=False)
        # points = torch.nonzero(to_point_mask[i], as_tuple=False)  ################
        # point = points[np.random.randint(len(points))]
        points = torch.nonzero(to_point_mask[i], as_tuple=False)
        if points.shape[0] == 0:
            # fallback：全零或随便在体积中取个点
            # 假设 prev_seg 形状是 (B,1,D,H,W)
            D, H, W = prev_seg.shape[-3], prev_seg.shape[-2], prev_seg.shape[-1]
            # 随机生成一个空间坐标 (c, z, y, x)
            # 这里把通道 c 固定为 0
            z = np.random.randint(D)
            y = np.random.randint(H)
            x = np.random.randint(W)
            point = torch.tensor([0, z, y, x], device=prev_seg.device)
            is_positive = False
        else:
            idx = np.random.randint(points.shape[0])
            point = points[idx]
            is_positive = bool(fn_masks[i, 0, point[1], point[2], point[3]])
        # print('point.shape',point.shape,'='*100)
        if fn_masks[i, 0, point[0], point[1], point[2]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    



def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):
        # # points = torch.argwhere(to_point_mask[i])
        # points = torch.nonzero(to_point_mask[i], as_tuple=False)
        # point = points[np.random.randint(len(points))]
        # # import pdb; pdb.set_trace()
        # if fn_masks[i, 0, point[1], point[2], point[3]]:
        #     is_positive = True
        # else:
        #     is_positive = False
        points = torch.nonzero(to_point_mask[i], as_tuple=False)
        if points.shape[0] == 0:
            # fallback：全零或随便在体积中取个点
            # 假设 prev_seg 形状是 (B,1,D,H,W)
            D, H, W = prev_seg.shape[-3], prev_seg.shape[-2], prev_seg.shape[-1]
            # 随机生成一个空间坐标 (c, z, y, x)
            # 这里把通道 c 固定为 0
            z = np.random.randint(D)
            y = np.random.randint(H)
            x = np.random.randint(W)
            point = torch.tensor([0, z, y, x], device=prev_seg.device)
            is_positive = False
        else:
            idx = np.random.randint(points.shape[0])
            point = points[idx]
            is_positive = bool(fn_masks[i, 0, point[1], point[2], point[3]])

        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    


def get_next_click3D_torch_with_dice(prev_seg, gt_semantic_seg):

    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        # mask_gt = mask_gt.astype(bool)
        mask_gt = (mask_gt > 0)
        
        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2*volume_intersect / volume_sum

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)


    for i in range(gt_semantic_seg.shape[0]):

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # bp = torch.tensor(point[1:]).reshape(1,1,3) 
        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    return batch_points, batch_labels, (sum(dice_list)/len(dice_list)).item()    


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_point(point, label, ax):
    if label == 0:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='red'))
    else:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='green'))
    # plt.scatter(point[0], point[1], label=label)

def get_next_click2D_ritm(prev_seg, gt_semantic_seg, used_regions=None):
    """
    改进版 RITM 点击策略，支持区域排除：
    - prev_seg: Tensor [B, 1, H, W] 或 [B, H, W]
    - gt_semantic_seg: Tensor [B, 1, H, W] 或 [B, H, W]  
    - used_regions: list 长度 B，每个元素是该batch已使用的区域mask [H, W]的列表
    
    返回:
    - batch_points: list 长度 B，每个元素 shape [N_i, 2] （N_i=0 或 1）
    - batch_labels: list 长度 B，每个元素 shape [N_i] （N_i=0 或 1）
    - new_used_regions: list 长度 B，更新后的已使用区域
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import ndimage
    
    try:
        import edt
    except ImportError:
        raise RuntimeError("需要安装 `edt` 库才能做距离变换")
    
    # 统一到 [B, H, W]
    assert prev_seg.device == gt_semantic_seg.device
    device = prev_seg.device
    
    if prev_seg.ndim == 4 and prev_seg.shape[1] == 1:
        prev_seg = prev_seg.squeeze(1)
    if gt_semantic_seg.ndim == 4 and gt_semantic_seg.shape[1] == 1:
        gt_semantic_seg = gt_semantic_seg.squeeze(1)
        
    B, H, W = prev_seg.shape
    mask_threshold = 0.5
    
    # 初始化已使用区域
    if used_regions is None:
        used_regions = [[] for _ in range(B)]
    
    # 转到 CPU
    pred_masks_cpu = (prev_seg > mask_threshold).to(torch.uint8).cpu().numpy()
    true_masks_cpu = (gt_semantic_seg > 0).to(torch.uint8).cpu().numpy()
    
    batch_points = []
    batch_labels = []
    new_used_regions = []
    
    for i in range(B):
        fn_i = (true_masks_cpu[i] == 1) & (pred_masks_cpu[i] == 0)  # FN区域
        fp_i = (true_masks_cpu[i] == 0) & (pred_masks_cpu[i] == 1)  # FP区域
        
        # 排除已使用的区域
        for used_mask in used_regions[i]:
            fn_i = fn_i & (~used_mask)
            fp_i = fp_i & (~used_mask)
        
        # 如果没有可用区域了
        if not fn_i.any() and not fp_i.any():
            batch_points.append(torch.empty((0, 2), device=device, dtype=torch.float32))
            batch_labels.append(torch.empty((0,), device=device, dtype=torch.int64))
            new_used_regions.append(used_regions[i].copy())
            continue
            
        # 找到连通区域并按大小排序
        best_region_mask, is_positive = find_best_unused_region(fn_i, fp_i)
        
        if best_region_mask is None:
            batch_points.append(torch.empty((0, 2), device=device, dtype=torch.float32))
            batch_labels.append(torch.empty((0,), device=device, dtype=torch.int64))
            new_used_regions.append(used_regions[i].copy())
            continue
        
        # 在最佳区域中选点
        coords_i, label_i = select_point_in_region(best_region_mask, is_positive, device)
        
        batch_points.append(coords_i)
        batch_labels.append(label_i)
        
        # 记录已使用的区域
        updated_used = used_regions[i].copy()
        updated_used.append(best_region_mask)
        new_used_regions.append(updated_used)
    
    return batch_points, batch_labels, new_used_regions


def find_best_unused_region(fn_mask, fp_mask):
    """
    找到FN或FP中最大的连通区域
    返回: (region_mask, is_positive) 或 (None, None)
    """
    from scipy import ndimage
    
    best_mask = None
    best_size = 0
    is_positive = True
    
    # 检查FN区域
    if fn_mask.any():
        fn_labeled, fn_num = ndimage.label(fn_mask)
        for label_id in range(1, fn_num + 1):
            region_mask = (fn_labeled == label_id)
            size = region_mask.sum()
            if size > best_size:
                best_size = size
                best_mask = region_mask
                is_positive = True
    
    # 检查FP区域  
    if fp_mask.any():
        fp_labeled, fp_num = ndimage.label(fp_mask)
        for label_id in range(1, fp_num + 1):
            region_mask = (fp_labeled == label_id)
            size = region_mask.sum()
            if size > best_size:
                best_size = size
                best_mask = region_mask
                is_positive = False
                
    return best_mask, is_positive


def select_point_in_region(region_mask, is_positive, device):
    """
    在给定区域中选择最优点
    """
    import edt
    
    # 对区域做距离变换
    region_pad = np.pad(region_mask.astype(np.uint8), pad_width=1, mode='constant', constant_values=0)
    region_dt = edt.edt(region_pad, black_border=True, parallel=4)[1:-1, 1:-1]
    
    # 找距离变换的最大值点（区域中心）
    max_val = region_dt.max()
    max_positions = np.argwhere(region_dt == max_val)
    
    # 如果有多个最大值点，随机选一个
    if len(max_positions) > 0:
        idx = np.random.randint(len(max_positions))
        y, x = max_positions[idx]
    else:
        # 备选方案：随机选择区域内任一点
        positions = np.argwhere(region_mask)
        idx = np.random.randint(len(positions))
        y, x = positions[idx]
    
    coords = torch.tensor([[x, y]], device=device, dtype=torch.float32)
    label = torch.tensor([int(is_positive)], device=device, dtype=torch.int64)
    
    return coords, label








if __name__ == "__main__":
    gt2D = torch.randn((2,1,256, 256)).cuda()
    prev_masks = torch.zeros_like(gt2D).to(gt2D.device)
    batch_points, batch_labels = get_next_click3D_torch(prev_masks.to(gt2D.device), gt2D)
    print(batch_points)