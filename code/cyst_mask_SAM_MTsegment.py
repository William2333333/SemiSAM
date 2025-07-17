import os
import pickle
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import numpy as np

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

current_dir = os.getcwd()
sam2_path = os.path.abspath(os.path.join(current_dir, 'sam2'))
if sam2_path not in sys.path:
    sys.path.append(sam2_path)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.click_method import get_next_click2D_ritm



from utils import ramps, losses
from dataloaders.Dataset2D import AlbSegDataset, TwoStreamBatchSampler



from utils.click_method import get_next_click2D_ritm
import albumentations as A
from albumentations.pytorch import ToTensorV2



parser = argparse.ArgumentParser()
parser.add_argument('--max_iterations', type=int,  default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.005, help='maximum epoch number to train')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=10, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=50000.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')  
args = parser.parse_args()








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
        images_dir = os.path.join(IMAGES_BASE_DIR, f"exam_{pid}", kidney_type)
        masks_dir  = os.path.join(MASKS_BASE_DIR,  f"exam_{pid}", kidney_type)

        if not os.path.isdir(images_dir):
            continue
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
                m_p = None   

            image_paths.append(img_p)
            mask_paths.append(m_p)

    return image_paths, mask_paths

def process_patient(labeled_ids, unlabeled_ids,train = True):
    kidneys = ["LK","RK"]
    dataset = {}
    if train:
        for k in kidneys:
            img_lab, msk_lab = process_kidney(labeled_ids,   k, with_mask=True)
            img_unl, msk_unl = process_kidney(unlabeled_ids, k, with_mask=False)
            dataset[k] = {
                "images": img_lab + img_unl,
                "masks":  msk_lab + msk_unl, 
              }  
    else:
        for k in kidneys:
            img_lab, msk_lab = process_kidney(labeled_ids,   k, with_mask=True)
            dataset[k] = {
                "images": img_lab,
                "masks":  msk_lab, 
              } 
            
    return dataset

def finetune_model_predict2D(
    imgs2D: torch.Tensor,     
    gts2D: torch.Tensor,     
    predictor,  
    device: str = "cuda",
    click_method=None,       
    num_clicks=3,
    prev_masks: torch.Tensor = None
) -> torch.Tensor:
    """

    """
    B, C, H, W = imgs2D.shape
    if gts2D.ndim == 3:
        gts2D = gts2D.unsqueeze(1)
    
    pred_batch = torch.zeros((B, 1, H, W), dtype=torch.uint8, device=device)

    for b in range(B):
        img_np = (imgs2D[b:b+1].cpu().numpy() * 255).astype("uint8")
        img_np = img_np.squeeze(0)               
        if img_np.shape[0] == 1:
            img_np = np.repeat(img_np, 3, axis=0)  
        img_np = img_np.transpose(1,2,0)          
        predictor.set_image(img_np)

        if prev_masks is None:
            cur_mask = torch.zeros((1,1,H,W),
                                   dtype=torch.uint8,
                                   device=device)
        else:
            cur_mask = prev_masks[b:b+1].clone()   
        acc_mask = cur_mask.clone()
        
        used_regions = None

        for _ in range(num_clicks):
            prev_2d = acc_mask.squeeze(1)
            gt_2d = gts2D[b:b+1].squeeze(1)
            pts, lbs, used_regions = click_method(prev_2d, gt_2d, used_regions)
            if all(len(points) == 0 for points in pts):
                break
            pts = torch.cat(pts, dim=0).cpu().numpy()
            lbs = torch.cat(lbs, dim=0).cpu().numpy()

            masks, scores, logits = predictor.predict(
                point_coords=pts,
                point_labels=lbs,
                multimask_output=False
            )
            cur_mask = (torch.from_numpy(masks[0].astype("uint8"))
                            .unsqueeze(0).unsqueeze(0)
                            .to(device))
            acc_mask = (acc_mask | cur_mask)
        pred_batch[b] = acc_mask       


    return pred_batch.to(torch.float32)




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
    # DATASET_1   train_1:2 * 1A+1B+1C+1D+1E (UNLABELED: 45 CASES) val_1:1B+1B
    labeled_patient_ids = ["010005-01","012438-01","010066-01","010087-01","010186-01","019233-02"]

    unlabeled_patient_ids = ["010880-02","010873-01","010187-02","012908-01",
                            "128249-01","012743-01","018899-01","012981-01",
                            "013044-01","018896-01","018900-01","012531-01",
                            "019286-02","019233-01","010896-01"
                            "010880-01","012532-01","018314-01","010044-01","010157-02","010179-02",
                            "010843-02","011905-01","018133-01","012583-01","018306-01","019605-01",
                            "019119-01","019605-02","012664-01"
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

    base_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.Normalize(mean=(0.0,0.0,0.0),std = (1.0,1.0,1.0),max_pixel_value = 255.0),
        ToTensorV2()
    ])

    semisup_ds_tmp = AlbSegDataset(
        semisup_train_dataset,
        image_transform=base_transform, 
        mask_transform=base_transform,
    )
    tmp_loader = DataLoader(
        semisup_ds_tmp,
        batch_sampler=sampler, 
        num_workers=4,         
        pin_memory=True,
    )

    sum_c    = torch.zeros(3, dtype=torch.double)
    sumsq_c  = torch.zeros(3, dtype=torch.double)
    num_pix  = 0
    for imgs, _ in tmp_loader:
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

    train_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),  
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

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

    model.train()
    ema_model.train()



# #############################################################################
    
    
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    criterion_first = nn.CrossEntropyLoss()  
    val_criterion = nn.CrossEntropyLoss()

    
    writer = SummaryWriter(snapshot_path+'/log')


            ########  Load SAM ########
    
    checkpoint_path = "../ckpt/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l"
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)

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
            loss_ce   = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs],weight=None)

            loss_dice = losses.dice_loss(outputs_soft[:labeled_bs,1],
                                        (label_batch[:labeled_bs] == 1).float())
            loss = (loss_ce + loss_dice)

            # MT-MODEL

            
            T = 8
            B, C, H, W = volume_batch[labeled_bs:].shape
            num_classes = 2

            imgs_r = volume_batch[labeled_bs:].repeat(2, 1, 1, 1)    
            stride = imgs_r.shape[0] // 2      

            preds = torch.zeros([stride * T, num_classes, H, W],device=device, dtype=torch.float32)

            ema_model.train()
            with torch.no_grad():
                for i in range(T // 2):
                    noisy = imgs_r + torch.clamp(torch.randn_like(imgs_r) * 0.1, -0.2, 0.2)

                    logits = ema_model(noisy)                  
                    start = 2 * stride * i
                    end   = 2 * stride * (i + 1)
                    preds[start:end] = logits

            preds = F.softmax(preds, dim=1)                    

            preds = preds.reshape(T, stride, num_classes, H, W)

            teacher_avg = preds.mean(dim=0)                  

            consistency_weight     = get_current_consistency_weight(iter_num)
            # MT-MODEL
            consistency_dist = F.mse_loss(
                outputs_soft[labeled_bs:],       
                teacher_avg         
            )

            consistency_loss = consistency_weight * consistency_dist
            loss += consistency_loss

        
            samseg_mask = finetune_model_predict2D(
            volume_batch[:labeled_bs], label_batch[:labeled_bs], predictor, device=device,
            click_method= get_next_click2D_ritm, num_clicks=10, 
            prev_masks=(outputs_soft[:,1:2] > 0.5).to(torch.uint8)) 
            samseg_soft = torch.cat([
            1 - samseg_mask,   
            samseg_mask        
            ], dim=1)

            sam_gt_dice_loss = losses.dice_loss(samseg_soft[:labeled_bs,1],(label_batch[:labeled_bs]==1).float())
            sam_gt_dice = 1.0 - sam_gt_dice_loss

            consistency_weight_sam = get_current_consistency_weight((max_iterations - iter_num))
            sam_consistency = F.mse_loss(
                outputs_soft[:labeled_bs],       
                samseg_soft[:labeled_bs]         
            )

            sam_con_loss =  consistency_weight_sam * sam_consistency



            loss += sam_con_loss    

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
            writer.add_scalar('loss/sam_con_loss', sam_con_loss, iter_num)

            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('train/sam_gt_dice', sam_gt_dice, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, sam_gt_dice: %f pre_gt_dice: %f' %
                    (iter_num, loss.item(), consistency_dist.item(), sam_gt_dice, 1.0-loss_dice))
            if iter_num % 100 == 0:
                
                img_l   = volume_batch[0]                      
                mask_prob = (outputs_soft[0,1] > 0.5).float()
                mask_t = (mask_prob.unsqueeze(0).unsqueeze(0)) 
                pred_vis = mask_t.repeat(1,3,1,1)             
                gt_l    = label_batch[0]                      

                import torchvision.utils as vutils

                img_l_norm = img_l.unsqueeze(0)          
                mean = torch.tensor(mean, device=img_l_norm.device).view(1,3,1,1)
                std  = torch.tensor(std,  device=img_l_norm.device).view(1,3,1,1)
                images_denorm = img_l_norm * std + mean
                images_denorm = images_denorm.clamp(0,1)  
                img_l_vis = images_denorm

                gt_rgb   = gt_l.unsqueeze(0).float()       
                gt_vis   = gt_rgb.repeat(1,3,1,1)           

                sam_mask = (samseg_soft[0:1,1]>0.5).float()
                sam_vis = sam_mask.repeat(1,3,1,1)

                grid = vutils.make_grid(
                    torch.cat([img_l_vis, pred_vis, gt_vis,sam_vis], dim=0), 
                    nrow=4, 
                    padding=2, 
                    normalize=False
                )  

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
            if iter_num % 50 == 0:
                model.eval()
                ema_model.eval()
                with torch.no_grad():
                    stu_val_loss_total = 0.0
                    stu_val_dice_total = 0.0
                    tea_val_loss_total = 0.0
                    tea_val_dice_total = 0.0
                    consistency_loss_total = 0.0
                    sam_con_loss_total = 0.0
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

                        consistency_dist_batch = F.mse_loss(
                            stu_probs,       
                            tea_probs         
                        )
                        consistency_weight_batch     = get_current_consistency_weight(iter_num)

                        consistency_loss_batch = consistency_weight_batch * consistency_dist_batch
                        consistency_loss_total += consistency_loss_batch.item()

                        samseg_mask_batch = finetune_model_predict2D(
                                val_imgs, val_masks, predictor, device=device,
                                click_method= get_next_click2D_ritm, num_clicks=10, 
                                prev_masks=(stu_probs[:,1:2] > 0.5).to(torch.uint8)) 
                        samseg_soft_batch = torch.cat([
                                1 - samseg_mask_batch,   
                                samseg_mask_batch        
                                ], dim=1)

                        sam_gt_dice_loss = losses.dice_loss(samseg_soft[:labeled_bs,1],(label_batch[:labeled_bs]==1).float())
                        sam_gt_dice = 1.0 - sam_gt_dice_loss

                        consistency_weight_sam_batch = get_current_consistency_weight((max_iterations - iter_num))
                        sam_consistency_batch = F.mse_loss(
                            stu_probs,       
                            samseg_soft_batch      
                        )

                        sam_con_loss_batch =  consistency_weight_sam_batch * sam_consistency_batch

                        sam_con_loss_total += sam_con_loss_batch




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
                    consistency_avg_val_loss = consistency_loss_total / max(val_batches, 1)
                    sam_con_avg_val_loss = sam_con_loss_total / max(val_batches, 1)
                    writer.add_scalar('val/tea_loss', tea_avg_val_loss, iter_num)
                    writer.add_scalar('val/tea_dice', tea_avg_val_dice, iter_num)
                    writer.add_scalar('val/stu_loss', stu_avg_val_loss, iter_num)
                    writer.add_scalar('val/stu_dice', stu_avg_val_dice, iter_num)
                    writer.add_scalar('val/consistency_loss', consistency_avg_val_loss, iter_num)
                    writer.add_scalar('val/sam_con_loss', sam_con_avg_val_loss, iter_num)



                model.train()
                
            if iter_num % 1000 == 0:
                tea_save_mode_path = os.path.join(snapshot_path, 'tea_semisup(weight10_lr0.005)_iter_' + str(iter_num) + '.pth')
                torch.save(ema_model.state_dict(), tea_save_mode_path)
                logging.info("save tea_model to {}".format(tea_save_mode_path))




            
            
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'tea_semisup(weight10_lr0.005)_iter_'+str(max_iterations)+'.pth')
    torch.save(ema_model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
