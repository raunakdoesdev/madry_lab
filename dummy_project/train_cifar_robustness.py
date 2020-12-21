import torch
from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
from robustness.defaults import check_and_fill_args
from robustness.model_utils import make_and_restore_model
import torch.nn.functional as F
from robustness.tools import helpers
from cox import utils
from tqdm.auto import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class WhiteboxPGD(torch.nn.Module):
    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)

    def calc_loss(self, x, target):
        """Get average loss for the batch (and apply normalization)"""
        return torch.mean(F.cross_entropy(self.model(self.normalize(x)), target))

    def clip(self, adv, img, eps):
        """L infinity clip"""
        return torch.clamp(torch.min(torch.max(adv, img - eps), img + eps), 0.0, 1.0)

    def forward(self, inp, target, eps, step_size, iterations, **kwargs):
        adv = inp + step_size * torch.rand_like(inp)
        for _ in range(iterations):
            adv = adv.clone().detach().requires_grad_(True)
            loss = self.calc_loss(adv, target)
            loss.backward()
            adv = self.clip(adv + step_size * torch.sign(adv.grad.data), inp, eps)  # gradient ASCENT
        return adv.clone().detach()


ds = CIFAR('/scratch/raunakc/datasets/cifar10')
model, _ = make_and_restore_model(arch='resnet18', dataset=ds)
model.attacker = WhiteboxPGD(model.model, ds)

train_kwargs = {
    'dataset': 'cifar',
    'arch': 'resnet',
    'out_dir': "train_out",
    'adv_train': 1,
    'adv_eval': 1,
    'eps': 8 / 255,
    'attack_lr': 2 / 255,
    'attack_steps': 10,
    'constraint': 'inf'  # not required but arg checker requires it :(
}

args = utils.Parameters(train_kwargs)
args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds.__class__)
if args.adv_train or args.adv_eval:
    args = check_and_fill_args(args, defaults.PGD_ARGS, ds.__class__)
args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds.__class__)

train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
train.train_model(args, model, (train_loader, val_loader))
