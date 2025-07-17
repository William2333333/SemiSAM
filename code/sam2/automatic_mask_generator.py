import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import skimage
from PIL import Image


input_dir = '/home/liao/code/aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized'
output_dir = '/home/liao/code/aimarcs-SAM/SAM/test_SAM/SAM_2_ORI_second_inference/Hiera_l'
os.makedirs(output_dir,exist_ok= True)

patient_id = "010005-01"
sam2_checkpoint = "/home/liao/code/sam2/checkpoints/sam2.1_hiera_large.pt"
#model_cfg = "/home/liao/code/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
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






def show_anns(orig_img,anns, borders=False, output_dir='', image_path=''):
    if len(anns) == 0:
        return
    image_path = image_path[:-4]
    mask_output_dir = os.path.join(output_dir, image_path)
    os.makedirs(mask_output_dir, exist_ok=True)

        # 创建一个新的画布，先显示原始图像
    fig, ax = plt.subplots()
    ax.imshow(orig_img)
    ax.axis('off')

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    combined_mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)

    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']


        combined_mask[m > 0] = 1

        # 保存每个单独的掩码

        mask_img = (m * 255).astype(np.uint8)
        output_path = os.path.join(mask_output_dir, f"mask_{i}.png")
        cv2.imwrite(output_path, mask_img)

    print("Saved masks in:", mask_output_dir)
    #ax.imshow(img)
    combined_mask_save_path = os.path.join(output_dir, f"{image_path}_combined.png")
    Image.fromarray((combined_mask * 255).astype(np.uint8)).save(combined_mask_save_path)
    print(f"Combined mask saved at: {combined_mask_save_path}")

    source_img = orig_img
    boundary_img = skimage.segmentation.mark_boundaries(source_img,combined_mask)

    fig, axs = plt.subplots(1,2,figsize= (12,6))

    axs[0].imshow(boundary_img)
    axs[0].set_title(f"{patient_id}_{image_path}_Boundary Image")
    axs[0].axis("off")

    axs[1].imshow(combined_mask, cmap = "gray")
    axs[1].set_title(f"{patient_id}_{image_path}_Combined Mask")
    axs[1].axis("off")

    plt.tight_layout()

    boundary_and_mask_img_save_path = os.path.join(output_dir, f"{image_path}_boundary_and_mask.png")
    plt.savefig(boundary_and_mask_img_save_path, bbox_inches = "tight")
    plt.show()


    print(f"Saved side by side image at: {boundary_and_mask_img_save_path}")



patient_path = f'/home/liao/code/aimarcs-SAM/SAM/data/samples_preprocessed/interpolated_images_resized/exam_{patient_id}'
patient_folder = os.path.basename(patient_path)
output_dir = os.path.join(output_dir, patient_folder)

for subpatient_folder in os.listdir(patient_path):
            subpatient_path = os.path.join(patient_path, subpatient_folder)
            if os.path.isdir(subpatient_path):
                subpatient_output_path = os.path.join(output_dir, subpatient_folder)
                os.makedirs(subpatient_output_path, exist_ok=True)

                for image_name in os.listdir(subpatient_path):
                    if image_name.endswith(".png"):
                        image_path = os.path.join(subpatient_path, image_name)
                        image = cv2.imread(image_path)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        masks = mask_generator.generate(image_rgb)
                        print(f"Generated {len(masks)} masks for {image_name}")

                        show_anns(orig_img=image_rgb,anns=masks, output_dir=subpatient_output_path, image_path=image_name)


