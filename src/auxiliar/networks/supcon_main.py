# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from supcon import SupCon

def main():
    
    # get device
    device = torch.device("cuda")
    
    # initialize backbone (resnet50)
    backbone = torchvision.models.resnet50(pretrained=True)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()


    # initialize ssl method
    model = SupCon(backbone, feature_size, projection_dim=32, temperature=0.07)
    model = model.to(device)
    transformers = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((32,32)),
    transforms.ToTensor()
    ])
    # load fake CIFAR-like dataset
    dataset = datasets.ImageFolder('data/supcon', transform=transformers) #datasets.FakeData(2000, (3, 32, 32), 10, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # switch to train mode
    model.train()
    
    # epoch training
    for epoch in range(10):
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
    
            # zero the parameter gradients
            model.zero_grad()
    
            # compute loss
            loss = model(images,labels)
            print('[Epoch %2d, Batch %2d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            
if __name__ == "__main__":
    main()