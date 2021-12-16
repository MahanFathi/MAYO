import copy
import random
from functools import wraps

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))

class AutoAugment(nn.Module):
    def __init__(self, output_size, in_channel):
        super().__init__()
        self.output_size = output_size
        self.in_channel = in_channel

        self.deconv_ops = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, 5, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.deconv_ops(x)
        x = F.interpolate(x, size=128, mode="bilinear")
        x += 1.
        x *= 255. / 2.
        return x


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.repr = {}
        self.hook_registered = False

    def _find_layer(self, layer):
        if type(layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(layer, None)
        elif type(layer) == int:
            children = [*self.net.children()]
            return children[layer]
        return None

    def _hook_hidden(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output

    def _hook_repr(self, _, input, output):
        device = input[0].device
        self.repr[device] = output

    def _register_hook(self):
        layer = self._find_layer(self.layer)
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook_hidden)

        layer = self._find_layer(self.layer + 1)
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook_repr)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, repr):
        dim = np.prod(repr.shape[1:])
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(repr)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        self.repr.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        repr = self.repr[x.device]
        self.hidden.clear()
        self.repr.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden, repr

    def forward(self, x, return_projection = True):
        hidden, representation = self.get_representation(x)

        if not return_projection:
            return hidden, representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, hidden, representation

# main class

class MAYO(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        eta = 1e-2,
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.crop_resize_augmentation = T.RandomResizedCrop((image_size, image_size))

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)
        self.autoaugment = AutoAugment(image_size, 512)

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.eta = eta

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

    def calc_norm_values(self, train_dataset):
        train_dataset = torch.tensor(train_dataset, dtype=torch.float32)
        norm_mean = torch.mean(train_dataset, dim=[-1, -2, 0])
        norm_std = torch.std(train_dataset, dim=[-1, -2, 0])
        self.norm_fn = T.Normalize(norm_mean, norm_std)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)[-1]

        x = self.crop_resize_augmentation(x)
        x = self.norm_fn(x)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj, target_hidden, target_repr = target_encoder(x)
            target_proj.detach_()
            target_hidden.detach_()
            target_repr.detach_()

        augmented_x = self.autoaugment(target_hidden.detach())
        augmented_x = self.norm_fn(augmented_x)
        online_proj, _, _ = self.online_encoder(augmented_x)
        online_pred = self.online_predictor(online_proj)

        loss = loss_fn(online_pred, target_proj.detach())
        loss += self.eta * F.mse_loss(x, augmented_x)

        return loss.mean()
