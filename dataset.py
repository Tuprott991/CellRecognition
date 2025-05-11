from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

class LIVECellSegDataset(Dataset):
    def __init__(self, image_dir, ann_path, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(ann_path)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, img_id + '.png')
        mask_path = os.path.join(self.masks_dir, img_id + '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # áp dụng riêng
        mask = (mask > 0).float()
        return image, mask
