## PyTorch
import torch
import torch.utils.data as data
# Torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# PyTorch Lightning
import lightning.pytorch as pl

from deep_autoencoder_aux import DATASET_PATH, download_data_cifar, embed_imgs, find_similar_images, get_train_images, show_reconstructions, show_test_mse, show_train_latent_dim

def setup():
    download_data_cifar()
    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def show_model_similarity(model, train_loader, test_loader):
    
    train_img_embeds = embed_imgs(model, train_loader)
    test_img_embeds = embed_imgs(model, test_loader)
    
    # Plot the closest images for the first N test images as example
    for i in range(8):
        find_similar_images(test_img_embeds[0][i], test_img_embeds[1][i], key_embeds=train_img_embeds)


def main():
    device = setup()

    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_set, val_set = torch.utils.data.random_split(CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True), [45000, 5000])
    #show_test_mse(train_set)
    # Loading the test set    
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)
    
    
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

    model_dict = show_train_latent_dim(device, train_loader, val_loader, test_loader, [64, 128, 256, 384])

    input_imgs = get_train_images(train_loader.dataset, 4)
    for latent_dim in model_dict:
        show_reconstructions(model_dict[latent_dim]["model"], input_imgs)

    # We use the following model throughout this section.
    # If you want to try a different latent dimensionality, change it here!
    model = model_dict[128]["model"]

    show_model_similarity(model)

if __name__ == '__main__':
    main()