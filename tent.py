from copy import deepcopy
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.jit
import time
import numpy as np



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # what would change it I don't force it, but only set module to train?
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    # return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


def train_model(
          model,
          dataloader,
          dataset_sizes,
          criterion,
          optimizer,
          device,
          scheduler = None,
          early_stopping = None,
          only_label = False,
          num_epochs = 30,
          label_amount = 0,
          alpha = 1,
          verbose=False):
    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_loss = np.Inf
    save_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()   # Set model to evaluate mode
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
                with torch.set_grad_enabled(phase == 'train'): # Double checked that this only enable BN layer param grad
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # calculate the losses
                    tent_loss = softmax_entropy(outputs).mean(0)
                    if label_amount == 0:
                        loss = tent_loss
                    else:
                        ce_loss = criterion(outputs[:label_amount,], labels[:label_amount,])
                        if only_label:
                            loss = ce_loss
                        else:
                            loss = alpha * ce_loss + tent_loss
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
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_acc = epoch_acc
                if early_stopping:
                    # early_stopping needs the validation loss to check if it has decresed,
                    early_stopping(epoch_loss, model)
                    # If the early stopping counter is 0, it means the val loss decreased
                    # deep copy the model
                    if early_stopping.counter == 0:
                        best_model_wts = deepcopy(model.state_dict())
                    if early_stopping.early_stop:
                        time_elapsed = time.time() - since
                        print('Training complete in {:.0f}m {:.0f}s'.format(
                            time_elapsed // 60, time_elapsed % 60))
                        print('Saved val Acc: {:4f}'.format(save_acc))
                        # load best model weights
                        model.load_state_dict(best_model_wts)
                        return model
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Saved val Acc: {:4f}'.format(save_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
