from torch import nn
import torchvision.models as models
import torch
import boto3


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_original_model_from_s3(device):
    # Create an S3 client
    s3 = boto3.client('s3')

    # List objects in a bucket
    bucket_name = 'nazar-models'

    # response = s3.list_objects_v2(Bucket=bucket_name)
    #
    # # Print object names
    # if 'Contents' in response:
    #     for obj in response['Contents']:
    #         print(obj['Key'])
    # else:
    #     print('The bucket is empty')

    s3.download_file(bucket_name, "nazar_original_model.pt", "nazar_original_model.pt")

    checkpoint = torch.load("nazar_original_model.pt")

    # for now this is useless...
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model = getattr(models, 'resnet50')()
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 201)
        model.load_state_dict(
            {k.replace('.layer.', '.'): v for k, v in checkpoint.items()})

    if torch.cuda.device_count() > 1:
        model = MyDataParallel(model)

    return model.to(device).eval()


def upload_new_model_to_s3(new_model_file_name):
    s3 = boto3.resource('s3')
    bucket_name = 'nazar-models'
    s3.Bucket(bucket_name).upload_file(new_model_file_name, new_model_file_name)
