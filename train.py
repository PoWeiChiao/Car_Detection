import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import SSD300, MultiBoxLoss
from utils.dataset import BDDDataset
from utils.logger import Logger
from utils.utils import *

def train(net, device, dataset_train, dataset_val, batch_size=16, epochs=150, lr=1e-3):
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True, collate_fn=dataset_val.collate_fn)

    optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(priors_cxcy=net.priors_cxcy).to(device=device)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    train_log = Logger('saved/train_log.txt')
    val_log = Logger('saved/val_log.txt')

    best_loss = float('inf')
    writer = SummaryWriter('runs/exp_2')

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        print('running epoch: {}'.format(epoch))
        net.train()
        for images, boxes, labels in tqdm(train_loader):
            images = images.to(device=device)
            boxes = [b.to(device=device) for b in boxes]
            labels = [l.to(device=device) for l in labels]

            predicted_locs, predicted_scores = net(images)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            train_loss += loss.item() * images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        for images, boxes, labels in tqdm(train_loader):
            images = images.to(device=device)
            boxes = [b.to(device=device) for b in boxes]
            labels = [l.to(device=device) for l in labels]
            with torch.no_grad():
                predicted_locs, predicted_scores = net(images)
                loss = criterion(predicted_locs, predicted_scores, boxes, labels)
                val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        lr_scheduler.step(val_loss)

        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, val_loss))
        writer.add_scalar('train loss', train_loss, epoch)
        writer.add_scalar('val loss', val_loss, epoch)
        train_log.write_epoch_loss(epoch, train_loss)
        val_log.write_epoch_loss(epoch, val_loss)
        if epoch >= 120:
            torch.save(net.state_dict(), 'saved/model_{}.pth'.format(epoch))
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), 'saved/model_best.pth')
            print('best model saved')

    writer.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    net = SSD300(n_classes=4)
    if os.path.isfile('saved/model_best.pth'):
        net.load_state_dict(torch.load('saved/model_best.pth', map_location=device))
    net.to(device=device)

    train_dir = 'data/train'
    val_dir = 'data/val'
    target_labels = ['car', 'truck', 'bus']
    target_size = [300, 300]
    train_image_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train = BDDDataset(data_dir=train_dir, target_labels=target_labels, target_size=target_size, image_transforms=train_image_transforms, is_random_transforms=True)
    dataset_val = BDDDataset(data_dir=val_dir, target_labels=target_labels, target_size=target_size, image_transforms=val_image_transforms)

    train(net, device, dataset_train, dataset_val)

if __name__ == '__main__':
    main()