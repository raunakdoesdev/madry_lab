import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import Adam
from torchvision.models import resnet34
import torch.nn.functional as F
import multiprocessing as mp
from pytorch_lightning.loggers import WandbLogger
import wandb


def get_dataloaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/data/theory/robustopt/datasets/cifar10', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=mp.cpu_count())

    testset = torchvision.datasets.CIFAR10(root='/data/theory/robustopt/datasets/cifar10', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=mp.cpu_count())

    return trainloader, testloader


class Classifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.1):
        super().__init__()
        self.hparams.lr = lr  # dummy value
        self.model = resnet34(pretrained=False, num_classes=num_classes)
        self.train_loader, self.val_loader = get_dataloaders()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.model(x)
        _, y_pred = torch.max(y_out, 1)
        loss = F.cross_entropy(y_out, y)
        self.log_dict({'train_acc': accuracy(y_pred, y), 'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.model(x)
        _, y_pred = torch.max(y_out, 1)
        self.log_dict({'val_loss': F.cross_entropy(y_out, y),
                       'val_acc': accuracy(y_pred, y)})

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


if __name__ == '__main__':
    lr = 1e-4
    logger = WandbLogger(experiment=wandb.init(project='madry_dummy_proj'))
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=3,
                                          filename='{epoch:02d}-{val_acc:.2f}')
    trainer = pl.Trainer(gpus=1, default_root_dir='/scratch/raunakc/madry-lab/dummy_project/checkpoints', logger=logger,
                         callbacks=[checkpoint_callback])
    if lr is None:
        model = Classifier()
        # Automatically find best learning rate (this is kinda iffy, safer to generate plot and pick the point urself)
        lr_finder = trainer.tuner.lr_find(model)
        wandb.log({'lrplot': lr_finder.plot(suggest=True)})
        model.hparams.lr = lr_finder.suggestion()
    else:
        model = Classifier(lr=lr)

    trainer.fit(model)
