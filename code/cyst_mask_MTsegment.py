import os
import pickle
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.utils as vutils

current_dir = os.getcwd()
pkdia_path = os.path.abspath(os.path.join(current_dir, '..', 'pkdia'))
if pkdia_path not in sys.path:
    sys.path.append(pkdia_path)
from nets import whichnet
current_dir = os.getcwd()


from utils import ramps, losses
from dataloaders.Dataset2D import AlbSegDataset, TwoStreamBatchSampler



import albumentations as A
from albumentations.pytorch import ToTensorV2



parser = argparse.ArgumentParser()
parser.add_argument('--max_iterations', type=int,  default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="kl", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=50000.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')  ######
args = parser.parse_args()






def criterion_consistency(p1, p2):
    return torch.mean((p1 - p2) ** 2)

def get_current_consistency_weight(step):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(step, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1- alpha)

def process_kidney(patient_ids, kidney_type, with_mask=True):
    image_paths = []
    mask_paths  = []

    for pid in patient_ids:
        # 你原先写的是 exam 前缀，注意拼接要一致
        images_dir = os.path.join(IMAGES_BASE_DIR, f"exam_{pid}", kidney_type)
        masks_dir  = os.path.join(MASKS_BASE_DIR,  f"exam_{pid}", kidney_type)

        if not os.path.isdir(images_dir):
            continue
        # 带标签才需要验证 masks_dir
        if with_mask and not os.path.isdir(masks_dir):
            continue

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(".png"):
                continue
            img_p = os.path.join(images_dir, fname)
            if with_mask:
                mask_fname = os.path.splitext(fname)[0] + "_combined.png"
                m_p = os.path.join(masks_dir, mask_fname)
                if not os.path.isfile(m_p):
                    continue
            else:
                m_p = None   # 用 None 代表“无标签”

            image_paths.append(img_p)
            mask_paths.append(m_p)

    return image_paths, mask_paths

def process_patient(labeled_ids, unlabeled_ids,train = True):
    kidneys = ["LK","RK"]
    dataset = {}
    if train:
        for k in kidneys:
            # 带标签
            img_lab, msk_lab = process_kidney(labeled_ids,   k, with_mask=True)
            # 无标签
            img_unl, msk_unl = process_kidney(unlabeled_ids, k, with_mask=False)
            # 把两者拼到一起
            dataset[k] = {
                "images": img_lab + img_unl,
                "masks":  msk_lab + msk_unl, 
              }  # msk_unl 中是 None
    else:
        for k in kidneys:
            # 带标签
            img_lab, msk_lab = process_kidney(labeled_ids,   k, with_mask=True)
            dataset[k] = {
                "images": img_lab,
                "masks":  msk_lab, 
              }  # msk_unl 中是 None
            
    return dataset


if __name__ == "__main__":
    snapshot_path = "../model/SemiSAM-MT-2/"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = args.device 


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size * len(args.gpu.split(','))
    max_iterations = args.max_iterations
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs

    #########################################################
    # Construct Dataset pkl file
    #########################################################
    BASE_DIR               = os.getcwd()
    IMAGES_BASE_DIR        = os.path.join(BASE_DIR, "../../aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized")
    MASKS_BASE_DIR         = os.path.join(BASE_DIR, "../../aimarcs-SAM/SAM/test_SAM/SAM_2_ORI_second_inference/Hiera_l")
    DATASET_SAVE_BASE_DIR  = os.path.join(BASE_DIR, "../data")
    # DATASET_1   train_1:1A+1B+1C val_1:1D+1E
    labeled_patient_ids = ["010005-01","012438-01","010066-01","010087-01","010186-01","019233-02"]

    unlabeled_patient_ids = ["010880-02","010873-01","010187-02","012908-01",
                            "128249-01","012743-01","018899-01","012981-01",
                            "013044-01","018896-01","018900-01","012531-01",
                            "019286-02","019233-01","010896-01",
                            "010880-01","012532-01","018314-01","010044-01","010157-02","010179-02",
                            "010843-02","011905-01","018133-01","012583-01","018306-01","019605-01",
                            "019119-01","019605-02","012664-01",
                            "010044-02","010869-03","018188-01","010885-01","010871-01","010174-01",
                            "010170-01","010869-02","010892-01","010065-02","010135-01","010073-01",
                            "010898-03","010064-01","120985-01"]
    
    validation_patient_ids = ["019287-01","019288-01"]

    # DATASET_2  train_2:train_1+val_1 val_2:test
    # labeled_patient_ids = ["010005-01","012438-01","010066-01","019287-01",
    #                     "019288-01","010087-01","010186-01","019233-02"]

    # unlabeled_patient_ids = ["010880-02","010873-01","010187-02","012908-01",
    #                         "128249-01","012743-01","018899-01","012981-01",
    #                         "013044-01","018896-01","018900-01","012531-01",
    #                         "019286-02","019233-01","010896-01"]
    
    # validation_patient_ids = ["012450-01", "012727-01", "010173-01", "012905-01", "018309-01"]

    os.makedirs(DATASET_SAVE_BASE_DIR, exist_ok=True)

    semisup_train_ds = process_patient(labeled_patient_ids,
                                unlabeled_patient_ids)
    semisup_train_pkl = os.path.join(DATASET_SAVE_BASE_DIR, "semisup_train_dataset.pkl")
    with open(semisup_train_pkl, "wb") as f:
        pickle.dump(semisup_train_ds, f)
    print("Saved semisup_train_dataset with both labeled & unlabeled to", semisup_train_pkl)


    validate_ds = process_patient(validation_patient_ids,unlabeled_ids=None,train=False)
    validate_pkl = os.path.join(DATASET_SAVE_BASE_DIR, "validate_dataset.pkl")
    with open(validate_pkl, "wb") as f:
        pickle.dump(validate_ds, f)
    print("Saved validate_dataset with labeled to", validate_pkl)






    #########################################################
    # Load Data
    #########################################################

    with open(semisup_train_pkl,"rb") as f:
        semisup_train_dataset = pickle.load(f)

    with open(validate_pkl,"rb") as f:
        validate_dataset = pickle.load(f)

    # 2) 遍历 LK/RK，统计,
    semisup_train_summary = {}
    for kidney in ["LK", "RK"]:
        imgs = semisup_train_dataset[kidney]["images"]
        masks = semisup_train_dataset[kidney]["masks"]
        total = len(imgs)
        labeled   = sum(1 for m in masks if m is not None)
        unlabeled = total - labeled
        semisup_train_summary[kidney] = {
            "total": total,
            "labeled": labeled,
            "unlabeled": unlabeled
        }

    print("semisup_train_dataset_Per-kidney slice counts:")
    for k, v in semisup_train_summary.items():
        print(f"  {k}:  total={v['total']}, "
            f"labeled={v['labeled']}, unlabeled={v['unlabeled']}")



    validate_summary = {}
    for kidney in ["LK", "RK"]:
        imgs = validate_dataset[kidney]["images"]
        masks = validate_dataset[kidney]["masks"]
        total = len(imgs)
        labeled   = sum(1 for m in masks if m is not None)
        unlabeled = total - labeled
        validate_summary[kidney] = {
            "total": total,
            "labeled": labeled,
            "unlabeled": unlabeled
        }

    print("validate_dataset_Per-kidney slice counts:")
    for k, v in validate_summary.items():
        print(f"  {k}:  total={v['total']}, "
            f"labeled={v['labeled']}, unlabeled={v['unlabeled']}")
    # 3) 如果你想把整个训练集看成一个序列，以构建 TwoStreamBatchSampler：,先把所有索引排列到一个 list,
    all_images = []
    all_masks  = []
    for kidney in ["LK","RK"]:
        all_images += semisup_train_dataset[kidney]["images"]
        all_masks  += semisup_train_dataset[kidney]["masks"]

    # primary（有标签）和 secondary（无标签）索引,
    primary_idxs   = [i for i,m in enumerate(all_masks) if m is not None]
    secondary_idxs = [i for i,m in enumerate(all_masks) if m is None]

    print(f"\nTotal slices: {len(all_images)}")
    print(f"  Primary (labeled) count:   {len(primary_idxs)}")
    print(f"  Secondary (unlabeled) count: {len(secondary_idxs)}")
    sampler = TwoStreamBatchSampler(primary_idxs,
    secondary_idxs,
    batch_size = batch_size,
    secondary_batch_size=batch_size-labeled_bs)

   # 1) 先定义一个只做 Resize+ToTensor 的临时 transform，用它来跑一次 DataLoader 计算 mean/std
    base_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.Normalize(mean=(0.0,0.0,0.0),std = (1.0,1.0,1.0),max_pixel_value = 255.0),
        ToTensorV2()
    ])

    semisup_ds_tmp = AlbSegDataset(
        semisup_train_dataset,
        image_transform=base_transform,  # 只 resize+to tensor
        mask_transform=base_transform,
    )
    tmp_loader = DataLoader(
        semisup_ds_tmp,
        batch_sampler=sampler,  # 保持和训练时一样的 sampler
        num_workers=4,          # 不要超过机器建议值
        pin_memory=True,
    )

    # 2) 累加求 mean 和 std
    sum_c    = torch.zeros(3, dtype=torch.double)
    sumsq_c  = torch.zeros(3, dtype=torch.double)
    num_pix  = 0
    for imgs, _ in tmp_loader:
        # imgs: [B,3,H,W], float32 in [0,1]
        # print(f"imgs shape: {imgs.shape}, dtype: {imgs.dtype}, device: {imgs.device}")
        # print(f"    min: {imgs.min().item():.4f}, max: {imgs.max().item():.4f}")
        imgs = imgs.double()
        B, C, H, W = imgs.shape
        sum_c   += imgs.sum(dim=[0,2,3])
        sumsq_c += (imgs**2).sum(dim=[0,2,3])
        num_pix += B * H * W

    mean = (sum_c / num_pix).float().tolist()
    var  = (sumsq_c / num_pix - (torch.tensor(mean, dtype=torch.double)**2)).clamp(min=0).float()
    std  = torch.sqrt(var).tolist()

    print("Dataset mean:", mean)
    print("Dataset std: ", std)

    # 3) 用计算好的 mean/std 重建 train/val transform
    train_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),  # ±10°旋转，50%概率
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # 4) 最后用新 transform 创建 DataLoader
    semisup_trian_dataset_trans = AlbSegDataset(
        semisup_train_dataset,
        image_transform=train_transform,
        mask_transform=train_transform,
    )
    semisup_loader = DataLoader(
        semisup_trian_dataset_trans,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    validate_dataset_trans = AlbSegDataset(
        validate_dataset,
        image_transform=val_transform,
        mask_transform=val_transform,
    )
    validate_loader = DataLoader(
        validate_dataset_trans,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


    #########################################################
    # Network
    #########################################################

    def create_model(ema=False):
            # Network definition
            net,_ = whichnet.whichnet(net_id = 1,n_channels = 3,n_classes = 2)
            model = net.cuda()
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model

    model = create_model()
    ema_model = create_model(ema=True)

    # student_ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/stu_just_unet_iter_2500.pth"   # 把这里改成你实际的文件路径
    # if os.path.isfile(student_ckpt_path):
    #     logging.info(f"Loading student U-Net checkpoint from {student_ckpt_path} …")
    #     ckpt = torch.load(student_ckpt_path, map_location=device)
    #     model.load_state_dict(ckpt)
    #     ema_model.load_state_dict(ckpt)
    #     logging.info("Loaded student checkpoint successfully.")
    # else:
    #     logging.warning(f"Cannot find checkpoint at {student_ckpt_path}, will train from scratch.")

    model.train()
    ema_model.train()



# #############################################################################
    
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_first = nn.CrossEntropyLoss()  # Multi-class segmentation loss
    val_criterion = nn.CrossEntropyLoss()

    
    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = 0
    max_epoch = max_iterations//len(semisup_loader)+1
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch)):
        
        for i_batch, sampled_batch in enumerate(semisup_loader):


            volume_batch, label_batch = sampled_batch[0], sampled_batch[1]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            if volume_batch.dtype == torch.uint8:
                    volume_batch = volume_batch.float() / 255.0
            outputs = model(volume_batch)
            outputs_soft = F.softmax(outputs, dim=1)
                    # 有监督分割损失
            # weight = torch.tensor([1.0,5.0]).cuda()
            loss_ce   = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs],weight=None)

            loss_dice = losses.dice_loss(outputs_soft[:labeled_bs,1],
                                        (label_batch[:labeled_bs] == 1).float())
            # loss = 0.5 * (loss_ce + loss_dice)
            loss = (loss_ce + loss_dice)

            # loss = criterion_first(outputs_soft[:labeled_bs], label_batch[:labeled_bs])
    
    
            # MT-MODEL

            
            T = 8
            B, C, H, W = volume_batch[labeled_bs:].shape
            num_classes = 2

            imgs_r = volume_batch[labeled_bs:].repeat(2, 1, 1, 1)     # [2B, C, H, W]
            stride = imgs_r.shape[0] // 2        # == B

            preds = torch.zeros([stride * T, num_classes, H, W],device=device, dtype=torch.float32)

            ema_model.train()
            with torch.no_grad():
                for i in range(T // 2):
                    # 在两倍 batch 上加噪
                    noisy = imgs_r + torch.clamp(torch.randn_like(imgs_r) * 0.1, -0.2, 0.2)
                    # 推理

                    logits = ema_model(noisy)                     # [2B, num_classes, H, W]
                    # 收集到 preds
                    start = 2 * stride * i
                    end   = 2 * stride * (i + 1)
                    preds[start:end] = logits

            preds = F.softmax(preds, dim=1)                     # [2B(T/2)=BT, num_classes, H, W]

            preds = preds.reshape(T, stride, num_classes, H, W)

            teacher_avg = preds.mean(dim=0)                     # [B, num_classes, H, W]

            # with torch.no_grad():
            #     preds_teacher = ema_model(volume_batch)

            # 超参数（前面代码里会计算好）
            consistency_weight     = get_current_consistency_weight(iter_num)
            # consistency_weight     = 0.8
            # MT-MODEL
            # # Mean-Teacher 一致性
            consistency_dist = F.mse_loss(
                outputs_soft[labeled_bs:],       # [B,2,H,W]
                teacher_avg         # [B,2,H,W]
            )
            # consistency_dist = F.mse_loss(
            #     outputs_soft,       # [B,2,H,W]
            #     F.softmax(preds_teacher, dim=1)         # [B,2,H,W]
            # )
            consistency_loss = consistency_weight * consistency_dist
            loss += consistency_loss



            # 5) 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            logging.info('iteration %d : loss : %f loss_ce: %f, loss_dice: %f ' %
                        (iter_num, loss.item(),loss_ce.item(),loss_dice.item()))
        
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, pre_gt_dice: %f' %
                    (iter_num, loss.item(), consistency_dist.item(), 1.0-loss_dice))
            
            if iter_num % 500 == 0:
                
                # —— 有标签样本（取 batch[0]） ——
                img_l   = volume_batch[0]                       # [1, H, W], float
                mask_prob = (outputs_soft[0,1] > 0.5).float()
                mask_t = (mask_prob.unsqueeze(0).unsqueeze(0))  # uint8 in {0,255}
                pred_vis = mask_t.repeat(1,3,1,1)               # back to float [0,1]
                gt_l    = label_batch[0]                        # [H, W], int

                import torchvision.utils as vutils

                img_l_norm = img_l.unsqueeze(0)            # [1, 1, H, W] or [1, 3, H, W]
                mean = torch.tensor(mean, device=img_l_norm.device).view(1,3,1,1)
                std  = torch.tensor(std,  device=img_l_norm.device).view(1,3,1,1)
                images_denorm = img_l_norm * std + mean
                images_denorm = images_denorm.clamp(0,1)  # [B,3,H,W]
                img_l_vis = images_denorm

                gt_rgb   = gt_l.unsqueeze(0).float()        # [1, H, W]
                gt_vis   = gt_rgb.repeat(1,3,1,1)           # [1,3,H,W]


                grid = vutils.make_grid(
                    torch.cat([img_l_vis, pred_vis, gt_vis], dim=0), 
                    nrow=3, 
                    padding=2, 
                    normalize=False
                )  # [3, H, 3W + padding2]

                print("pred_vis.shape, dtype, min, max:",
                        pred_vis.shape,
                        pred_vis.dtype,
                        pred_vis.min().item(),
                        pred_vis.max().item())

                writer.add_image('train/supervised_sam_MT', grid, iter_num)

                
#######################################################################################################################################################
            ## change lr
            # if iter_num % 2500 == 0:
            #     # lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     lr_ = base_lr * 0.5** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
                    
            # if iter_num % 10000 == 0:
            #     # lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     lr_ = base_lr * 0.5** (iter_num // 10000)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            if iter_num % 500 == 0:
                model.eval()
                ema_model.eval()
                with torch.no_grad():
                    stu_val_loss_total = 0.0
                    stu_val_dice_total = 0.0
                    tea_val_loss_total = 0.0
                    tea_val_dice_total = 0.0
                    val_batches = 0

                    for v_batch, v_sample in enumerate(validate_loader):
                        val_imgs, val_masks = v_sample[0], v_sample[1]  # val_imgs:[B,C,H,W]; val_masks:[B,H,W] or [B,1,H,W]
                        val_imgs = val_imgs.cuda()
                        val_masks = val_masks.cuda()
                        if val_masks.ndim == 4:
                            val_masks = val_masks.squeeze(1)  # -> [B, H, W]

                        # 前向：Teacher 和 Student
                        if val_imgs.dtype == torch.uint8:
                            val_imgs = val_imgs.float() / 255.0
                        tea_logits = ema_model(val_imgs)       # [B, 2, H, W]
                        tea_probs  = F.softmax(tea_logits, dim=1)
                        stu_logits = model(val_imgs)           # [B, 2, H, W]
                        stu_probs  = F.softmax(stu_logits, dim=1)

                        # 计算交叉熵 loss
                        tea_loss_ce = val_criterion(tea_logits, val_masks)
                        stu_loss_ce = val_criterion(stu_logits, val_masks)
                        tea_val_loss_total += tea_loss_ce.item()
                        stu_val_loss_total += stu_loss_ce.item()

                        # 计算 Dice coefficient
                        tea_pred_fg = tea_probs[:, 1, :, :]       # [B, H, W]
                        stu_pred_fg = stu_probs[:, 1, :, :]       # [B, H, W]
                        gt_fg_float = (val_masks == 1).float()    # [B, H, W]
                        tea_dice_loss_batch = losses.dice_loss(tea_pred_fg, gt_fg_float)
                        stu_dice_loss_batch = losses.dice_loss(stu_pred_fg, gt_fg_float)
                        tea_dice_coeff_batch = 1.0 - tea_dice_loss_batch
                        stu_dice_coeff_batch = 1.0 - stu_dice_loss_batch
                        tea_val_dice_total += tea_dice_coeff_batch.item()
                        stu_val_dice_total += stu_dice_coeff_batch.item()

                        # —— 把这个 batch 的“输入图 + 学生预测掩码 + 真实掩码”拼成一个 grid，写到 TensorBoard —— 
                        # val_imgs: [B, C, H, W]，假定 C=1 或 3；stu_pred_fg: [B, H, W]；val_masks: [B, H, W]
                        imgs_normalized = val_imgs.clone()  # TensorBoard 期待的输入范围通常在 [0,1]
                        if imgs_normalized.shape[1] == 1:
                            imgs_visual = imgs_normalized.repeat(1, 3, 1, 1)  # 单通道->三通道
                        else:
                            imgs_visual = imgs_normalized  # 本身就是 3 通道

                        mean = torch.tensor(mean, device=imgs_visual.device).view(1,3,1,1)
                        std  = torch.tensor(std, device=imgs_visual.device).view(1,3,1,1)
                        images_denorm = imgs_visual * std + mean
                        images_denorm = images_denorm.clamp(0,1)  # [B,3,H,W]

                        # 把学生预测二值化：>0.5 为前景
                        stu_pred_binary = (stu_pred_fg > 0.5).float().unsqueeze(1)  # [B,1,H,W]
                        stu_pred_vis = stu_pred_binary.repeat(1, 3, 1, 1)               # [B,3,H,W]
                        tea_pred_binary = (tea_pred_fg > 0.5).float().unsqueeze(1)  # [B,1,H,W]
                        tea_pred_vis = tea_pred_binary.repeat(1, 3, 1, 1)               # [B,3,H,W]


                        # 把 GT mask 转成 one-hot 三通道可视化（1->白色，0->黑色）
                        gt_binary = val_masks.unsqueeze(1).float()             # [B,1,H,W]
                        gt_vis = gt_binary.repeat(1, 3, 1, 1)                   # [B,3,H,W]

                        # 拼接： inputs | student_pred | gt_mask  -> 在宽度方向并排
                        # 最终得到 [B, 3, H, 3*W]
                        stu_concat = torch.cat([images_denorm, stu_pred_vis, gt_vis], dim=3)
                        tea_concat = torch.cat([images_denorm, tea_pred_vis, gt_vis], dim=3)


                        # 将这个 batch 的 concatenated tensor 写入 TensorBoard
                        # tag 里可以包含迭代次数和 batch idx，方便检索
                        writer.add_image(
                            f"val/student_pred",
                            vutils.make_grid(stu_concat, nrow=1, normalize=False),
                            global_step=iter_num
                        )

                        writer.add_image(
                            f"val/teacher_pred",
                            vutils.make_grid(tea_concat, nrow=1, normalize=False),
                            global_step=iter_num
                        )

                        val_batches += 1

                    # 计算并记录平均验证 loss 和 dice
                    tea_avg_val_loss = tea_val_loss_total / max(val_batches, 1)
                    tea_avg_val_dice = tea_val_dice_total / max(val_batches, 1)
                    stu_avg_val_loss = stu_val_loss_total / max(val_batches, 1)
                    stu_avg_val_dice = stu_val_dice_total / max(val_batches, 1)
                    writer.add_scalar('val/tea_loss', tea_avg_val_loss, iter_num)
                    writer.add_scalar('val/tea_dice', tea_avg_val_dice, iter_num)
                    writer.add_scalar('val/stu_loss', stu_avg_val_loss, iter_num)
                    writer.add_scalar('val/stu_dice', stu_avg_val_dice, iter_num)

                model.train()
                
            if iter_num % 500 == 0:
                # stu_save_mode_path = os.path.join(snapshot_path, 'stu_semisup(45unlabel)_iter_' + str(iter_num) + '.pth')
                tea_save_mode_path = os.path.join(snapshot_path, 'tea_semisup(45unlabel)_iter_' + str(iter_num) + '.pth')
                # torch.save(model.state_dict(), stu_save_mode_path)
                torch.save(ema_model.state_dict(), tea_save_mode_path)
                # logging.info("save stu_model to {}".format(stu_save_mode_path))
                logging.info("save tea_model to {}".format(tea_save_mode_path))




            
            
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'tea_semisup(45unlabel)_iter_'+str(max_iterations)+'.pth')
    torch.save(ema_model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
