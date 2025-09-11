import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

class INRIADataset(Dataset):
    def __init__(self, root, split='train', **kwargs):
        self.root = root
        self.split = split
        self.images = sorted(os.listdir(os.path.join(root, split, 'images')))
        self.masks = sorted(os.listdir(os.path.join(root, split, 'masks')))

        print(len(self.images))
        print(len(self.masks))

        # Agumentations
        # if split == 'train':
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),  
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.RandomGamma(p=0.2),
            A.ElasticTransform(p=0.1, alpha=120, sigma=6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1), 
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),  # Add geometric distortion
            A.Cutout(p=0.1, max_h_size=32, max_w_size=32, num_holes=5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        # else:
        #     self.transform = A.Compose([
        #         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #         ToTensorV2()
        #     ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root, self.split, 'images', self.images[idx])
        mask_path = os.path.join(self.root, self.split, 'masks', self.masks[idx])

        # Loading image and mask as numpy arrays
        image = np.array(Image.open(img_path).convert('RGB')).astype(np.uint8)
        # image dim: (H x W x 3)
        mask = np.load(mask_path)
        # mask dim: (H x W X 1)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)  # shape → (H, W, 1)


        # Apply Agumentations
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        # Ensure shape is (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = mask.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

        mask = mask.float()



        return {'image' : image, 'mask': mask}
