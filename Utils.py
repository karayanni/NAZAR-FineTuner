import random
from fine_tune import ImageFilelist
import torch
import os
import mysql.connector
from torchvision import transforms


def get_images_names_from_sql(fine_tune_by: dict):
    host_db_uri = 'freedriftlogdb.chgu9pxp8lci.us-east-1.rds.amazonaws.com'

    cnx = mysql.connector.connect(user='admin', password='nadernader',
                                  host=host_db_uri,
                                  database='drift_log_schema')

    # can be change to drifted_query = "" in case we want to tune on all images with atts.
    drifted_query = "counter_drift = 1"
    for key in fine_tune_by.keys():
        if drifted_query == "":
            drifted_query += key + " = '" + fine_tune_by[key] + "'"
        else:
            drifted_query += " AND " + key + " = '" + fine_tune_by[key] + "'"

    cursor = cnx.cursor()
    cursor.execute(
        "SELECT img_url FROM drift_log_schema.drift_log_flex where " + drifted_query)
    res = cursor.fetchall()

    cursor.close()
    cnx.close()

    img_list = []
    for item in res:
        img_list.append(item[0])

    return img_list


def loaders_from_imlist(fine_tune_by: dict):
    batch_size = 128
    num_workers = 4

    root = os.path.join(".", 'images')
    #
    # imlist = ['images_simulation/2020-01-01-bj_0-n02018795-0.png',
    #           'images_simulation/2020-01-01-bj_0-n02018795-1.png',
    #           'images_simulation/2020-01-01-bj_0-n02018795-2.png',
    #           'images_simulation/2020-01-01-bj_10-n01443537-2.png']

    imlist = get_images_names_from_sql(fine_tune_by)

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
