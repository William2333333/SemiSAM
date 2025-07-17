import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

input_dir = '/home/liao/code/aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized'
output_dir = '/home/liao/code/aimarcs-SAM/SAM/test_SAM/SAM_2_ORI/Hiera_l/preprocessed_unified'
#output_dir = '/home/liao/code/aimarcs-SAM/SAM/test_SAM/SAM_1_REFINE/ViT-L/preprocessed_unidfied'

sam2_checkpoint = "/home/liao/code/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "/home/liao/code/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

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
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
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
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)





np.random.seed(3)

def show_anns(anns, borders=True, output_dir=None, image_path=None):
    if len(anns) == 0:
        return
    image_path = image_path[:-4]
    output_dir = os.path.join(output_dir, image_path)
    os.makedirs(output_dir, exist_ok=True)

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 

        # 保存每个单独的掩码

        mask_img = (m * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, f"mask_{i}.png")
        cv2.imwrite(output_path, mask_img)

        if borders:
            
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
    print("Saved masks in:", output_dir)
    ax.imshow(img)
    overlay_save_path = os.path.join(output_dir, f"{image_path}_overlay.png")
    plt.imsave(overlay_save_path, img)
    print(f"Saved: {overlay_save_path}")

#patient_path = '/home/liao/code/aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized/exam_10066_1'
#patient_folder = os.path.basename(patient_path)
#output_dir = os.path.join(output_dir, patient_folder)

# 遍历输入基础文件夹下所有病例（patient 文件夹）
for patient_folder in os.listdir(input_dir):
    patient_path = os.path.join(input_dir, patient_folder)
    if os.path.isdir(patient_path):
        print(f"Processing patient: {patient_folder}")
        # 为每个病例创建对应的输出文件夹
        patient_output_dir = os.path.join(output_dir, patient_folder)
        os.makedirs(patient_output_dir, exist_ok=True)


        for subpatient_folder in os.listdir(patient_path):
            subpatient_path = os.path.join(patient_path, subpatient_folder)
            if os.path.isdir(subpatient_path):
                subpatient_output_path = os.path.join(patient_output_dir, subpatient_folder)
                os.makedirs(subpatient_output_path, exist_ok=True)

                for image_name in os.listdir(subpatient_path):
                    if image_name.endswith(".png"):
                        image_path = os.path.join(subpatient_path, image_name)
                        image = cv2.imread(image_path)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        masks = mask_generator.generate(image_rgb)
                        print(f"Generated {len(masks)} masks for {image_name}")

                        show_anns(masks, subpatient_output_path, image_name)


