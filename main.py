import os
import torch
import threading
from fine_tune import load_model, ImageFilelist, train_model, collect_params, configure_model, EarlyStopping
import torchvision.models as models
from tqdm import tqdm
from scipy.special import softmax
from torch import nn
from torchvision import transforms
from Utils import loaders_from_imlist
from s3utils import get_original_model_from_s3, upload_new_model_to_s3
from flask import Flask, jsonify, request

app = Flask("nazar-finetuner")


def get_ds():
    save_img_path = os.path.join(".", 'images')

    data_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TODO: change the images to be read form the cloud by cause and drift = True.
    ds = ImageFilelist(save_img_path, ['2020-01-02-uk_0-n02206856-0.png'], transform=data_transformer)
    return ds


def fine_tune(fine_tune_by: dict):
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = models.resnet50()
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    # model.fc = nn.Linear(2048, 201)
    # torch.save(model.state_dict(), file_path)

    # model = load_model(model_type="resnet50", checkpoint_path='model.pt', device=gpu_device)

    ds = get_ds()
    cls_pred = []
    drift = []
    model_ids = []

    # this below is only for getting the fine-tuning plan...
    #
    # for j in tqdm(range(len(ds))):
    #     (x, y) = ds[j]
    #     inputs = torch.unsqueeze(x, 0).to(gpu_device)
    #
    #     with torch.no_grad():
    #         # TODO: extract the below info form DB if needed
    #         location = 'Quebec'
    #         weather = 'clear-day'
    #         model_type = 'Resnet50'
    #         device_id = 'niceid22'
    #         # TODO: for each image inference using the most fitted model.
    #
    #         # TODO: switch this to getting the appropriate model.
    #         most_fitted_model = model
    #         model_id = 'original'
    #         # outputs, model_id = workload_simulator.inference_on_prompt((location, weather, model_type, device_id),
    #         #                                                            inputs)
    #
    #         # this is the inference
    #         outputs = most_fitted_model(inputs)
    #
    #     outputs = outputs.cpu()
    #     _, pred = torch.max(outputs, 1)
    #     cls_pred.append(pred[0].item() == y)
    #     drift.append((max(softmax(outputs, 1)[0]) < 0.9).item())
    #     model_ids.append(model_id)

    # for any cause, we retrain the original model.

    # tmp_model = load_model(model_type="resnet50", checkpoint_path='model.pt', device=gpu_device)

    tmp_model = get_original_model_from_s3(gpu_device)

    configure_model(tmp_model, 'fc')
    params, _ = collect_params(tmp_model, 'fc')

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    lr = 0.1
    momentum = 0.9
    step_size = 7
    gamma = 0.1
    patience = 7
    num_epochs = 30
    label_amount = 0
    alpha = 1.0

    optimizer_ft = torch.optim.SGD(params, lr=lr, momentum=momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size, gamma)
    # initialize the early_stopping object
    early_stop = EarlyStopping(patience, verbose=True)
    train_parameters = {'criterion': criterion,
                        'optimizer': optimizer_ft,
                        'scheduler': exp_lr_scheduler,
                        'num_epochs': num_epochs,
                        'early_stopping': early_stop,
                        'label_amount': label_amount,
                        'alpha': alpha,
                        'mode': 'tent'
                        }

    ds_loader = loaders_from_imlist()

    tmp_model = train_model(tmp_model, ds_loader, train_parameters=train_parameters, verbose=True).eval()

    new_model_file_name = '_'.join(fine_tune_by.values()) + '_' + 'tent' + '.pt'
    path_to_new_model = os.path.join('.', new_model_file_name)

    torch.save(tmp_model.state_dict(), path_to_new_model)

    upload_new_model_to_s3(new_model_file_name)

    print("fine tuning job completed")

@app.route('/finetune')
def hello_world():
    data = request.get_json()

    fine_tuning_plan = {}
    for key, value in data.items():
        fine_tuning_plan[key] = value

    print("starting fine tuning with plan:")
    print(fine_tuning_plan)

    threading.Thread(target=fine_tune, args=(fine_tuning_plan,)).start()

    return jsonify(f'Fine Tuned job started to tune the original model with regards to {", ".join(fine_tuning_plan.values())}')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
