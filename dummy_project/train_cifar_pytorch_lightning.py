import sys

sys.path.append('..')

import multiprocessing as mp

import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy

from dummy_project.resnet import *


def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(),
    ])

    transform_val = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='/scratch/raunakc/datasets/cifar10', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, pin_memory=True,
                                              shuffle=True, num_workers=mp.cpu_count())

    testset = torchvision.datasets.CIFAR10(root='/scratch/raunakc/datasets/cifar10', train=False,
                                           download=True, transform=transform_val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, pin_memory=True,
                                             shuffle=False, num_workers=mp.cpu_count())

    return trainloader, testloader


class Classifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.1):
        super().__init__()
        self.hparams.lr = lr
        self.model = ResNet101()
        self.train_loader, self.val_loader = get_dataloaders()

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.model(x)
        _, y_pred = torch.max(y_out, 1)
        loss = F.cross_entropy(y_out, y)
        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.model(x)
        _, y_pred = torch.max(y_out, 1)
        self.val_acc(y_pred, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return {'optimizer': optimizer, 'scheduler': scheduler}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


if __name__ == '__main__':
    lr = 1e-1
    logger = WandbLogger(experiment=wandb.init(project='madry_dummy_proj'))
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=3,
                                          filename='{epoch:02d}-{val_acc:.2f}')
    trainer = pl.Trainer(gpus=[2], default_root_dir='/scratch/raunakc/madry-lab/dummy_project/checkpoints',
                         logger=logger,
                         callbacks=[checkpoint_callback], auto_select_gpus=True)
    if lr is None:
        model = Classifier()
        # Automatically find best learning rate (this is kinda iffy, safer to generate plot and pick the point urself)
        lr_finder = trainer.tuner.lr_find(model)
        wandb.log({'lrplot': lr_finder.plot(suggest=True)})
        model.hparams.lr = lr_finder.suggestion()
    else:
        model = Classifier(lr=lr)

    trainer.fit(model)
