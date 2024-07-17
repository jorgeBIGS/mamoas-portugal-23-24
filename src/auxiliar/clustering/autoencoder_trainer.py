from lightning import Trainer
import torch
from torchvision import transforms
import torchvision.datasets as data 
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import lightning.pytorch as pl

from autoencoder import Decoder, Encoder, LitAutoEncoder

from autoencoders import *

DATASET_PATH = 'data/clustering/eurosat'
CHECKPOINTING_DIR = 'data/checkpointing'
NUM_WORKERS = 19
NUM_GPUS = 1
VAL_PARTITION = [0.6,0.4]
pl.seed_everything(42)

'''def load_CIFAR():
    width = 32
    height = 32
    num_bands = 3
    train_set, val_set = torch.utils.data.random_split(data.CIFAR(DATASET_PATH, train=True, 
                                                                  download=True, 
                                                                  transform=transforms.ToTensor()), VAL_PARTITION)
    test_set = data.MNIST(DATASET_PATH, train = False, download=True, transform=transforms.ToTensor())
    return train_set, val_set, test_set, width, height, num_bands'''

def load_MNIST():
    width = 28
    height = 28
    num_bands = 1
    train_set, val_set = torch.utils.data.random_split(data.MNIST(DATASET_PATH, train=True, 
                                                                  download=True, 
                                                                  transform=transforms.ToTensor()), VAL_PARTITION)
    test_set = data.MNIST(DATASET_PATH, train = False, download=True, transform=transforms.ToTensor())
    return train_set, val_set, test_set, width, height, num_bands

def train_model(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
    # model
    autoencoder = new_n_band_deep_autoencoder(32, 3)

    # train model
    trainer = Trainer(accelerator="gpu", devices=NUM_GPUS, default_root_dir=CHECKPOINTING_DIR)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(autoencoder, dataloaders=test_loader)

def main():
    train_set, val_set, test_set = load_MNIST()
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    train_model(train_loader, val_loader, test_loader, width, height)
    
    


if __name__ == '__main__':
    main()
    