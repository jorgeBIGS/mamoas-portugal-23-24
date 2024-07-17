import os
import torch.nn.functional as F
import torchvision
import torch

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

from auxiliar.clustering.deep_autoencoder import Autoencoder, GenerateCallback
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

import urllib.request
from urllib.error import HTTPError

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "tutoriales/data/cifar"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "tutoriales/saved_models/tutorial9"

def get_train_images(train_dataset, num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

def download_data_cifar():
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial9/"
    # Files to download
    pretrained_files = ["cifar10_64.ckpt", "cifar10_128.ckpt", "cifar10_256.ckpt", "cifar10_384.ckpt"]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)

def show_compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4,2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def show_test_mse(train_dataset):
    for i in range(2):
        # Load example image
        img, _ = train_dataset[i]
        img_mean = img.mean(dim=[1,2], keepdims=True)

        # Shift image by one pixel
        SHIFT = 1
        img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
        img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
        img_shifted[:,:1,:] = img_mean
        img_shifted[:,:,:1] = img_mean
        show_compare_imgs(img, img_shifted, "Shifted -")

        # Set half of the image to zero
        img_masked = img.clone()
        img_masked[:,:img_masked.shape[1]//2,:] = img_mean
        show_compare_imgs(img, img_masked, "Masked -")

def train_cifar(device, train_loader, val_loader, test_loader, latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(train_loader.dataset, 8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

def show_train_latent_dim(device, train_loader, val_loader, test_loader, dims):
    model_dict = {}
    for latent_dim in dims:
        model_ld, result_ld = train_cifar(device, train_loader, val_loader, test_loader, latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}

    latent_dims = sorted([k for k in model_dict])
    val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

    fig = plt.figure(figsize=(6,4))
    plt.plot(latent_dims, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.minorticks_off()
    plt.ylim(0,100)
    plt.show()

    return model_dict

def show_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    for imgs, _ in data_loader:
        with torch.no_grad():
            z = model.encoder(imgs.to(model.device))
        img_list.append(imgs)
        embed_list.append(z)
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))

def find_similar_images(query_img, query_z, key_embeds, K=8):
    # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
    dist = torch.cdist(query_z[None,:], key_embeds[1], p=2)
    dist = dist.squeeze(dim=0)
    dist, indices = torch.sort(dist)
    # Plot K closest images
    imgs_to_display = torch.cat([query_img[None], key_embeds[0][indices[:K]]], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_display, nrow=K+1, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12,3))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
