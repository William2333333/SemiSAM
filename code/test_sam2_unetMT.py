import os
import sys
import argparse
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from medpy import metric
from utils import losses
from medpy.metric.binary import jc as metric_jc, hd95 as metric_hd95, asd as metric_asd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from dataloaders.Dataset2D import AlbSegDataset


# Add pkdia path
current_dir = os.getcwd()
pkdia_path = os.path.abspath(os.path.join(current_dir, '..', 'pkdia'))
if pkdia_path not in sys.path:
    sys.path.append(pkdia_path)
from nets import whichnet

def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray):
    """Compute Dice, Jaccard, HD95, ASD for a single binary prediction."""
    dice = metric.binary.dc(pred, gt)
    jc   = metric.binary.jc(pred, gt)
    hd   = metric.binary.hd95(pred, gt)
    asd  = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd

def create_model(ema: bool = False):
    """Instantiate U-Net (or other) and optionally detach for EMA."""
    net, _ = whichnet.whichnet(net_id=1, n_channels=3, n_classes=2)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def process_kidney(patient_ids, kidney_type, with_mask=True):
    image_paths = []
    mask_paths  = []
    for pid in patient_ids:
        images_dir = os.path.join(IMAGES_BASE_DIR, f"exam_{pid}", kidney_type)
        masks_dir  = os.path.join(MASKS_BASE_DIR,  f"exam_{pid}", kidney_type)
        if not os.path.isdir(images_dir): continue
        if with_mask and not os.path.isdir(masks_dir): continue
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(".png"): continue
            img_p = os.path.join(images_dir, fname)
            if with_mask:
                mask_fname = os.path.splitext(fname)[0] + "_combined.png"
                m_p = os.path.join(masks_dir, mask_fname)
                if not os.path.isfile(m_p): continue
            else:
                m_p = None
            image_paths.append(img_p)
            mask_paths.append(m_p)
    return image_paths, mask_paths

def process_patient(labeled_ids, unlabeled_ids, train=True):
    kidneys = ["LK","RK"]
    dataset = {}
    if train:
        for k in kidneys:
            img_lab, msk_lab = process_kidney(labeled_ids,   k, with_mask=True)
            img_unl, msk_unl = process_kidney(unlabeled_ids, k, with_mask=False)
            dataset[k] = {"images": img_lab+img_unl, "masks": msk_lab+msk_unl}
    else:
        for k in kidneys:
            img_lab, msk_lab = process_kidney(labeled_ids, k, with_mask=True)
            dataset[k] = {"images": img_lab, "masks": msk_lab}
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/liao/code/SemiSAM/data')
    parser.add_argument('--model',     type=str, default='SemiSAM-MT-2')
    parser.add_argument('--gpu',       type=str, default='0')
    parser.add_argument('--epoch',     type=int, default=10000)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    snapshot_path  = f"../model/{args.model}/"
    test_save_path = "../model/prediction/sam2_unetMT/256_256_iter_10000_2D/"
    os.makedirs(test_save_path, exist_ok=True)

    global IMAGES_BASE_DIR, MASKS_BASE_DIR, DATASET_SAVE_BASE_DIR
    BASE_DIR              = os.getcwd()
    IMAGES_BASE_DIR       = os.path.join(BASE_DIR, "../../aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized")
    MASKS_BASE_DIR        = os.path.join(BASE_DIR, "../../aimarcs-SAM/SAM/test_SAM/SAM_2_ORI_second_inference/Hiera_l")
    DATASET_SAVE_BASE_DIR = os.path.join(BASE_DIR, "../data")

    # prepare test dataset
    # test dataset
    test_patient_ids = ["012450-01", "012727-01", "010173-01", "012905-01", "018309-01"]
    # 1A
    # test_patient_ids = ["012450-01"]
    # 1B
    # test_patient_ids = [ "012727-01"]
    # 1C
    # test_patient_ids = [ "010173-01"]
    # 1D
    # test_patient_ids = [ "012905-01"]
    # 1E
    # test_patient_ids = [ "018309-01"]

    test_ds = process_patient(test_patient_ids, unlabeled_ids=None, train=False)
    test_pkl = os.path.join(DATASET_SAVE_BASE_DIR, "test_dataset.pkl")
    with open(test_pkl, "wb") as f:
        pickle.dump(test_ds, f)

    # test transforms
    base_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.Normalize(mean=(0.0,0.0,0.0),std = (1.0,1.0,1.0),max_pixel_value = 255.0),
        ToTensorV2()
    ])


    with open(test_pkl, "rb") as f:
        test_dataset = pickle.load(f)

    # summarize test set
    print("test_dataset slice counts:")
    for k in ["LK","RK"]:
        imgs  = test_dataset[k]["images"]
        masks = test_dataset[k]["masks"]
        print(f"  {k}: total={len(imgs)}, labeled={sum(m is not None for m in masks)}")

    test_dataset_tmp = AlbSegDataset(test_dataset,
                                       image_transform=base_transform,
                                       mask_transform=base_transform)
    test_loader_tmp = DataLoader(test_dataset_tmp,
                             batch_size=32, shuffle=True,
                             num_workers=4, pin_memory=True)
    
    sum_c    = torch.zeros(3, dtype=torch.double)
    sumsq_c  = torch.zeros(3, dtype=torch.double)
    num_pix  = 0
    for imgs, _ in test_loader_tmp:

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
    test_transform = A.Compose([
        A.Resize(height=256, width=256, interpolation=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_dataset_trans = AlbSegDataset(
        test_dataset,
        image_transform=test_transform,
        mask_transform=test_transform,
    )




    # load model
    model = create_model()

    # train,val: working, test:train+val(not working)
    # ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/stu_checkpoint_semisup_noraug_iter_350.pth"
    # test:test(working)
    # ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/stu_just_unet_with_nor_aug_iter_3000.pth"
    # ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel)_iter_9500.pth"
    # MT15
    # ckpt_path_15 = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(15unlabel)_iter_30000.pth"
    # ckpt_path_30 = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(30unlabel)_iter_10500.pth"
    # ckpt_path_45 = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel)_iter_9000.pth"
    # ckpt_paths = [ckpt_path_15,ckpt_path_30,ckpt_path_45]
    # semisam_click3_45unlabel

    # ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel_click3)_iter_15000.pth"
    # semisam_click6_45unlabel

    # ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel_click6)_iter_10500.pth"
   
    # semisam_click10_45unlabel

    # ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel_click10)_iter_10500.pth"
    
    # semisam_click10_45unlabel_weight1.0

    ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel_click10_weight1)_iter_100000.pth"



    if os.path.isfile(ckpt_path):
        print(f"Loading checkpoint {ckpt_path} …")
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))

    else:
        print(f"Checkpoint not found at {ckpt_path}, abort.")
        return
    model.eval()


    # setup tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(snapshot_path, 'test_logs'))

    total_dice    = 0.0
    total_jc      = 0.0
    total_hd95    = 0.0
    total_asd     = 0.0
    total_batches = 0

    total_avg_dice    = 0.0
    total_avg_jc      = 0.0
    total_avg_hd95    = 0.0
    total_avg_asd     = 0.0


    with torch.no_grad():
        for i in range(5):
            test_loader = DataLoader(
                test_dataset_trans,
                batch_size=16,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            for t_batch, (imgs, masks) in enumerate(test_loader):
                imgs  = imgs.cuda()
                masks = masks.cuda()
                if masks.ndim==4: 
                    masks = masks.squeeze(1)

                if imgs.dtype == torch.uint8:
                        imgs = imgs.float() / 255.0
                logits = model(imgs)

                probs  = F.softmax(logits, dim=1)       # [B,2,H,W]
                preds  = torch.argmax(probs, dim=1)

                # -------- Soft Dice（训练一致） --------
                fg_probs = probs[:, 1, :, :]            # [B,H,W]
                gt_fg    = (masks == 1).float()         # [B,H,W]
                dice_loss_batch  = losses.dice_loss(fg_probs, gt_fg)
                dice_coeff_batch = (1.0 - dice_loss_batch).item()

                # -------- Hard Metrics (阈值 0.5) --------
                preds_np = (fg_probs > 0.5).cpu().numpy().astype(np.uint8)  # [B,H,W]
                gt_np    = masks.cpu().numpy().astype(np.uint8)            # [B,H,W]
                batch_jc   = []
                batch_hd95 = []
                batch_asd  = []
                B = preds_np.shape[0]
                for j in range(B):
                    p = preds_np[j]
                    g = gt_np[j]
                    # 如果 GT 全 0 且预测全 0，跳过 HD95/ASD 计算（或设为 0）
                    if g.sum() == 0 or p.sum() == 0:
                        batch_jc.append(1.0)
                        batch_hd95.append(0.0)
                        batch_asd.append(0.0)
                    else:
                        batch_jc.append( metric_jc(p, g) )
                        batch_hd95.append( metric_hd95(p, g) )
                        batch_asd.append( metric_asd(p, g) )
                    
                    

                jc_batch   = float(np.mean(batch_jc))
                hd95_batch = float(np.mean(batch_hd95))
                asd_batch  = float(np.mean(batch_asd))

                # 写入 TensorBoard
                writer.add_scalar('test/dice',  dice_coeff_batch, t_batch)
                writer.add_scalar('test/jc',    jc_batch,           t_batch)
                writer.add_scalar('test/hd95',  hd95_batch,         t_batch)
                writer.add_scalar('test/asd',   asd_batch,          t_batch)



                img_l_norm = imgs           # [1, 1, H, W] or [1, 3, H, W]
                mean = torch.tensor(mean, device=img_l_norm.device).view(1,3,1,1)
                std  = torch.tensor(std,  device=img_l_norm.device).view(1,3,1,1)
                images_denorm = img_l_norm * std + mean
                images_denorm = images_denorm.clamp(0,1)  # [B,3,H,W]
                img_l_vis = images_denorm
                gt_vis      = masks.unsqueeze(1).repeat(1,3,1,1).float()
                pr_vis      = preds.unsqueeze(1).repeat(1,3,1,1).float()
                concat      = torch.cat([img_l_vis, pr_vis, gt_vis], dim=3)
                grid        = make_grid(concat, nrow=1, normalize=False)
                writer.add_image('test/with_gt_and_pred', grid, t_batch)

                # 累加全局
                total_dice    += dice_coeff_batch
                total_jc      += jc_batch
                total_hd95    += hd95_batch
                total_asd     += asd_batch
                total_batches += 1


            avg_dice  = total_dice  / max(total_batches, 1)
            avg_jc    = total_jc    / max(total_batches, 1)
            avg_hd95  = total_hd95  / max(total_batches, 1)
            avg_asd   = total_asd   / max(total_batches, 1)

            total_avg_dice += avg_dice
            total_avg_jc += avg_jc
            total_avg_hd95 += avg_hd95
            total_avg_asd += avg_asd
    
    avg_dice = total_avg_dice / 5
    avg_jc = total_avg_jc / 5
    avg_hd95 = total_avg_hd95 / 5
    avg_asd = total_avg_asd / 5

    print("Overall Average Metrics on Test Set:")
    print(f"  Dice  : {avg_dice:.4f}")
    print(f"  Jaccard: {avg_jc:.4f}")
    print(f"  HD95  : {avg_hd95:.4f}")
    print(f"  ASD   : {avg_asd:.4f}")

    writer.add_scalar('test/avg_dice',  avg_dice)
    writer.add_scalar('test/avg_jc',    avg_jc)
    writer.add_scalar('test/avg_hd95',  avg_hd95)
    writer.add_scalar('test/avg_asd',   avg_asd)


    writer.close()
    print("Testing complete.")

if __name__ == '__main__':
    main()

