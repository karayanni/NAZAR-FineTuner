import os
import torch
import torchvision.models as models
import numpy as np
from torch import nn
import torch.utils.data as data
import re
from PIL import Image
from copy import deepcopy
import time
from tqdm import tqdm


class_to_idx = {"n01440764": 0, "n01443537": 1, "n01484850": 2, "n01491361": 3, "n01494475": 4, "n01496331": 5,
                "n01498041": 6,
                "n01530575": 7, "n01531178": 8, "n01532829": 9, "n01534433": 10, "n01537544": 11, "n01558993": 12,
                "n01560419": 13, "n01580077": 14, "n01582220": 15, "n01592084": 16, "n01601694": 17, "n01614925": 18,
                "n01616318": 19, "n01622779": 20, "n01630670": 21, "n01631663": 22, "n01632458": 23, "n01632777": 24,
                "n01641577": 25, "n01644373": 26, "n01644900": 27, "n01664065": 28, "n01665541": 29, "n01667114": 30,
                "n01667778": 31, "n01669191": 32, "n01675722": 33, "n01677366": 34, "n01682714": 35, "n01685808": 36,
                "n01687978": 37, "n01688243": 38, "n01689811": 39, "n01692333": 40, "n01693334": 41, "n01704323": 42,
                "n01728572": 43, "n01728920": 44, "n01729322": 45, "n01729977": 46, "n01734418": 47, "n01735189": 48,
                "n01737021": 49, "n01740131": 50, "n01744401": 51, "n01751748": 52, "n01755581": 53, "n01756291": 54,
                "n01768244": 55, "n01770081": 56, "n01770393": 57, "n01773157": 58, "n01773549": 59, "n01773797": 60,
                "n01774384": 61, "n01774750": 62, "n01775062": 63, "n01776313": 64, "n01784675": 65, "n01795545": 66,
                "n01796340": 67, "n01797886": 68, "n01798484": 69, "n01806567": 70, "n01807496": 71, "n01819313": 72,
                "n01820546": 73, "n01824575": 74, "n01828970": 75, "n01833805": 76, "n01847000": 77, "n01855032": 78,
                "n01855672": 79, "n01860187": 80, "n01871265": 81, "n01872401": 82, "n01873310": 83, "n01877812": 84,
                "n01882714": 85, "n01883070": 86, "n01910747": 87, "n01914609": 88, "n01917289": 89, "n01924916": 90,
                "n01930112": 91, "n01943899": 92, "n01944390": 93, "n01945685": 94, "n01950731": 95, "n01955084": 96,
                "n01968897": 97, "n01978287": 98, "n01978455": 99, "n01983481": 100, "n01985128": 101, "n01986214": 102,
                "n01990800": 103, "n02002724": 104, "n02006656": 105, "n02007558": 106, "n02009229": 107,
                "n02009912": 108,
                "n02011460": 109, "n02012849": 110, "n02017213": 111, "n02018207": 112, "n02018795": 113,
                "n02025239": 114,
                "n02027492": 115, "n02028035": 116, "n02033041": 117, "n02037110": 118, "n02051845": 119,
                "n02058221": 120,
                "n02066245": 121, "n02071294": 122, "n02074367": 123, "n02077923": 124, "n02114367": 125,
                "n02114548": 126,
                "n02114855": 127, "n02115641": 128, "n02115913": 129, "n02117135": 130, "n02119022": 131,
                "n02119789": 132,
                "n02120079": 133, "n02120505": 134, "n02125311": 135, "n02127052": 136, "n02128385": 137,
                "n02128757": 138,
                "n02132136": 139, "n02133161": 140, "n02134084": 141, "n02137549": 142, "n02165105": 143,
                "n02165456": 144,
                "n02167151": 145, "n02168699": 146, "n02169497": 147, "n02172182": 148, "n02174001": 149,
                "n02177972": 150,
                "n02190166": 151, "n02206856": 152, "n02219486": 153, "n02226429": 154, "n02229544": 155,
                "n02231487": 156,
                "n02233338": 157, "n02236044": 158, "n02259212": 159, "n02264363": 160, "n02268443": 161,
                "n02268853": 162,
                "n02276258": 163, "n02279972": 164, "n02280649": 165, "n02281406": 166, "n02281787": 167,
                "n02317335": 168,
                "n02319095": 169, "n02321529": 170, "n02325366": 171, "n02326432": 172, "n02346627": 173,
                "n02356798": 174,
                "n02361337": 175, "n02363005": 176, "n02389026": 177, "n02395406": 178, "n02396427": 179,
                "n02410509": 180,
                "n02412080": 181, "n02415577": 182, "n02437312": 183, "n02441942": 184, "n02442845": 185,
                "n02443114": 186,
                "n02444819": 187, "n02445715": 188, "n02447366": 189, "n02487347": 190, "n02488291": 191,
                "n02509815": 192,
                "n02514041": 193, "n02526121": 194, "n02536864": 195, "n02606052": 196, "n02640242": 197,
                "n02641379": 198,
                "n02643566": 199, "n02655020": 200}


@torch.jit.script
def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def load_model(model_type='Resnet50',
               checkpoint_path='model.pt',
               device=torch.device('cpu')):
    checkpoint = torch.load(checkpoint_path)
    # for now this is useless...
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model = getattr(models, model_type.lower())()
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 201)
        model.load_state_dict(
            {k.replace('.layer.', '.'): v for k, v in checkpoint.items()})
    return model.to(device).eval()


# dataset util
def label_seperator(filename):
    return class_to_idx[re.findall(r"([n]\d+)", filename)[0]]


def collect_params(model, config_type='bn'):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    if config_type == 'bn':
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    elif config_type == 'fc':
        for np, p in model.fc.named_parameters():
            if np in ['weight', 'bias']:  # weight is scale, bias is shift
                params.append(p)
                names.append(f"fc.{np}")
    else:
        for nm, m in model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def configure_model(model_param, config_type='bn'):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model_param.train()
    # disable grad, to (re-)enable only what tent updates
    model_param.requires_grad_(False)
    if config_type == 'bn':
        # configure norm for tent updates: enable grad + force batch statisics
        for m in model_param.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
    elif config_type == 'fc':
        model_param.fc.requires_grad_(True)
    else:
        model_param.requires_grad_(True)


class ImageFilelist(data.Dataset):
    def __init__(self, root, imlist, transform=None, labeler=label_seperator):
        self.root = root
        self.imlist = imlist
        self.size = len(imlist)
        self.transform = transform
        self.labeler = labeler

    def __getitem__(self, index):
        impath, label = self.imlist[index], self.labeler(self.imlist[index])
        img = Image.open(os.path.join(self.root, impath)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return torch.Tensor(img), label

    def __len__(self):
        return len(self.imlist)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.verbose = verbose

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model(model, dataloader, train_parameters, verbose=False):
    criterion, optimizer, scheduler, num_epochs, early_stopping, mode = \
        train_parameters['criterion'], train_parameters['optimizer'], train_parameters['scheduler'], \
            train_parameters['num_epochs'], train_parameters['early_stopping'], train_parameters['mode']
    assert mode in ['fine-tune', 'tent', 'tent_fc', 'tent_all', 'soft-pseudo-label', 'hard-pseudo-label', 'memo']

    if mode == 'fine-tune':
        best_score = 0.0
    else:
        label_amount, alpha = train_parameters['label_amount'], train_parameters['alpha']
        best_score = np.Inf

    if mode == 'tent_fc':
        w0 = deepcopy(model.fc.weight)
    if mode == 'tent_all':
        model_w0 = {}
        for nm, m in deepcopy(model).named_modules():
            for np_, p in m.named_parameters():
                if np_ in ['weight']:
                    model_w0[f"{nm}.{np_}"] = p

    device = next(model.parameters()).device
    dataset_sizes = {x: dataloader[x].dataset.size for x in ['train', 'val']}
    best_model_wts = deepcopy(model.state_dict())
    save_acc = 0.0

    since = time.time()
    for epoch in tqdm(range(num_epochs)):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # calculate the losses
                    loss = softmax_entropy(outputs).mean(0)
                    if label_amount > 0:
                        if len(labels) > label_amount:
                            loss += alpha * criterion(outputs[:label_amount, ], labels[:label_amount, ])
                        else:
                            loss += alpha * criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':

                if mode == 'fine-tune':
                    if epoch_acc > best_score:
                        best_score = epoch_acc
                        save_acc = epoch_acc
                        best_model_wts = deepcopy(model.state_dict())

                elif epoch_loss < best_score:
                    best_score = epoch_loss
                    save_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())

                if early_stopping:
                    # early_stopping needs the validation loss to check if it has decresed,
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping(epoch_loss, model)
                    if early_stopping.early_stop:
                        time_elapsed = time.time() - since
                        print('Training complete in {:.0f}m {:.0f}s'.format(
                            time_elapsed // 60, time_elapsed % 60))
                        print('Best val Acc: {:4f}'.format(save_acc))

                        # load best model weights
                        model.load_state_dict(best_model_wts)
                        return model

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
