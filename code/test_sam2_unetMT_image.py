import os
import sys
import argparse
import logging
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from medpy import metric
from utils import losses
from medpy.metric.binary import jc as metric_jc, hd95 as metric_hd95, asd as metric_asd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from dataloaders.Dataset2D import CustomDataset, AlbSegDataset


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
    # train and val dataset
    # test_patient_ids = ["010005-01","012438-01","010066-01","019287-01",
    #                     "019288-01","010087-01","010186-01","019233-02"]
    # validation dataset
    # test_patient_ids = ["010186-01","019233-02"]
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
    test_loader = DataLoader(
        test_dataset_trans,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )



    # load model
    ema_model = create_model(ema=True)
    model_15 = create_model()
    model_30 = create_model()
    model_45 = create_model()

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
    # # semisam_click1_45unlabel
    # # ckpt_path = "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_NOSAM_semisup(45unlabel)_iter_10500.pth"

    # for ckpt_path in ckpt_paths:
    #     if os.path.isfile(ckpt_path):
    #         print(f"Loading checkpoint {ckpt_path} …")
    #         model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))

    #     else:
    #         print(f"Checkpoint not found at {ckpt_path}, abort.")
    #         return
    # model.eval()



    # ckpt_paths = {
    #     'MT15': "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(15unlabel)_iter_30000.pth",
    #     'MT30': "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(30unlabel)_iter_10500.pth",
    #     'MT45': "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel)_iter_9000.pth"
    # }
    ckpt_paths = {
        'CK3': "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel_click3)_iter_15000.pth",
        'CK6': "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel_click6)_iter_10500.pth",
        'CK10': "/home/liao/code/SemiSAM/model/SemiSAM-MT-2/tea_semisup(45unlabel_click10)_iter_9500.pth"
    }
    models = {'CK3': model_15, 'CK6': model_30, 'CK10': model_45}
    for name, model in models.items():
        ckpt = ckpt_paths[name]
        if os.path.isfile(ckpt):
            print(f"Loading checkpoint for {name}: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model.eval()
    # setup tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(snapshot_path, 'test_logs'))

    # 全局累加器
    # total_metric_sum = np.zeros(4, dtype=np.float64)
    # total_samples    = 0
    total_dice    = 0.0
    total_jc      = 0.0
    total_hd95    = 0.0
    total_asd     = 0.0
    total_batches = 0


    with torch.no_grad():
        for t_batch, (imgs, masks) in enumerate(test_loader):
            imgs = imgs.cuda()
            masks = masks.cuda().squeeze(1)  # assume [B,1,H,W]
            mean = torch.tensor(mean, device=imgs.device).view(1,3,1,1)
            std  = torch.tensor(std,  device=imgs.device).view(1,3,1,1)
            # Denormalize images
            imgs_denorm = imgs * std + mean
            imgs_denorm = imgs_denorm.clamp(0,1)

            for name, model in models.items():
                # Forward pass
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                preds = (probs[:,1] > 0.5).float().unsqueeze(1)

                # Prepare tensors for visualization
                img_vis = imgs_denorm                # [B,3,H,W]
                pred_vis = preds.repeat(1,3,1,1)      # [B,3,H,W]
                gt_vis = masks.unsqueeze(1).repeat(1,3,1,1)  # [B,3,H,W]

                # Concatenate along width
                concat = torch.cat([img_vis, pred_vis, gt_vis], dim=3)  # [B,3,H,W*3]
                grid = make_grid(concat, nrow=1, normalize=False)

                # Write to TensorBoard
                writer.add_image(f"Test/{name}", grid, global_step=t_batch)

    writer.close()

if __name__ == '__main__':
    main()

