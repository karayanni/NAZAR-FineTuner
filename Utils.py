import random
from fine_tune import ImageFilelist
import torch
import os
from torchvision import transforms


def loaders_from_imlist():
    batch_size = 128
    num_workers = 4

    # TODO: change with images from DB with cause and drift...
    root = os.path.join(".", 'images')
    imlist = ['2020-01-02-uk_0-n02206856-0.png',
              '2020-01-02-uk_3-n02206856-1.png',
              '2020-01-02-uk_5-n02206856-1.png',
              '2020-01-02-uk_6-n01665541-2.png']
    split = {'train': 0.7, 'val': 0.3}
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ]), 'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])}
    shuffle = True
    assert split['train'] + split['val'] == 1
    if shuffle:
        random.seed(10)
        random.shuffle(imlist)

    train_ds = ImageFilelist(
        root, imlist[:int(len(imlist) * split['train'])], data_transforms['train'])
    val_ds = ImageFilelist(
        root, imlist[int(len(imlist) * split['train']):], data_transforms['val'])

    return {'train': torch.utils.data.DataLoader(train_ds,
                                                 batch_size=batch_size,
                                                 shuffle=False, drop_last=False, num_workers=num_workers),
            'val': torch.utils.data.DataLoader(val_ds,
                                               batch_size=batch_size,
                                               shuffle=False, drop_last=False, num_workers=num_workers)}
