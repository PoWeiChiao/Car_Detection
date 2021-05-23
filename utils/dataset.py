import glob
import json
import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BDDDataset(Dataset):
    def __init__(self, data_dir, target_labels, target_size, image_transforms, is_random_transforms=False):
        self.data_dir = data_dir
        self.target_labels = target_labels
        self.target_size = target_size
        self.image_transforms = image_transforms
        self.is_random_transforms = is_random_transforms

        self.images_list = glob.glob(os.path.join(data_dir, 'image', '*.jpg'))
        self.labels_list = glob.glob(os.path.join(data_dir, 'label', '*.json'))

        self.images_list.sort()
        self.labels_list.sort()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        isFlip = random.random()

        image = Image.open(self.images_list[idx])
        w, h = image.size
        if self.is_random_transforms and isFlip > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        scale_factor = [self.target_size[0] / w, self.target_size[1] / h]
        image = image.resize(self.target_size)

        f = open(self.labels_list[idx])
        label = json.load(f)
        boxes = []
        labels = []
        for obj in label['object']:
            for target in self.target_labels:
                if obj['class'] == target:
                    labels.append(self.target_labels.index(target) + 1)
                    x0 = obj['x1'] * scale_factor[0] / self.target_size[0]
                    y0 = obj['y1'] * scale_factor[1] / self.target_size[1]
                    x1 = obj['x2'] * scale_factor[0] / self.target_size[0]
                    y1 = obj['y2'] * scale_factor[1] / self.target_size[1]
                    if self.is_random_transforms and isFlip > 0.5:
                        x0 = (self.target_size[0] - (obj['x2'] * scale_factor[0])) / self.target_size[0]
                        x1 = (self.target_size[0] - (obj['x1'] * scale_factor[0])) / self.target_size[1]
                    boxes.append([x0, y0, x1, y1])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        f.close()

        image = self.image_transforms(image)
        return image, boxes, labels
    
    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels

def main():
    data_dir = 'D:/pytorch/ObjectDetection/BDD/data/train'
    target_labels = ['car', 'truck', 'bus']
    target_size = [300, 300]
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = BDDDataset(data_dir=data_dir, target_labels=target_labels, target_size=target_size, image_transforms=image_transforms)
    print(dataset.__len__())

if __name__ == '__main__':
    main()









