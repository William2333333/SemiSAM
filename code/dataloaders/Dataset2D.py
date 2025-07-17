from torch.utils.data import Dataset        
import numpy as np                          
import torch                                
import cv2                                  
from torch.utils.data.sampler import Sampler

class AlbSegDataset(Dataset):
    """
    A PyTorch Dataset for segmentation tasks that uses Albumentations for augmentations.

    Args:
        dataset_dict (dict): 
            Mapping with keys 'LK' and 'RK', each containing:
                'images': list of image file paths
                'masks':  list of mask file paths (or None for missing masks)
        image_transform (albumentations.Compose, optional):
            Compose object for image augmentations + normalization.
        mask_transform (albumentations.Compose, optional):
            Compose object for mask geometric transforms.
    """
    def __init__(self, dataset_dict, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform  = mask_transform

        # Flatten all image and mask paths into two lists
        self.image_paths = []
        self.mask_paths  = []
        for kidney in ["LK", "RK"]:
            # Extend the lists with paths from left kidney and right kidney
            self.image_paths.extend(dataset_dict[kidney]["images"])
            self.mask_paths .extend(dataset_dict[kidney]["masks"])

    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load an image and its corresponding mask, apply transforms, and return tensors.

        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            img (torch.Tensor): Transformed image tensor of shape [3, H, W], values in [0,1].
            mask (torch.Tensor): Binary mask tensor of shape [H, W], values {0,1}.
        """
        img_p = self.image_paths[idx]
        m_p   = self.mask_paths[idx]  # Mask path, may be None

        # 1) Read the image from disk (OpenCV loads in BGR)
        img = cv2.imread(img_p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # 2) Read the mask; if missing, create an all-zero mask
        if m_p is None:
            # Generate a zero mask with the same spatial dimensions as the image
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Load mask as single-channel grayscale (values 0 or 255)
            mask = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)

        # 3) Apply Albumentations transforms to both image and mask, if provided
        if self.image_transform and self.mask_transform:
            augmented = self.image_transform(image=img, mask=mask)
            img  = augmented["image"]  # Returns a Tensor of shape [3, H, W]
            mask = augmented["mask"]   # Returns a Tensor of shape [H, W], values still {0,255}
        else:
            # If no transforms are provided, convert to tensors directly
            from torchvision.transforms import functional as TF
            img  = TF.to_tensor(img)           # Converts to FloatTensor [0,1]
            mask = torch.from_numpy(mask).long()

        # 4) Convert mask values from {0,255} to binary {0,1}
        mask = (mask > 0).long()

        return img, mask
    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

