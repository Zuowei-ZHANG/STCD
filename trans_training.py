import os
import time
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from net import CDCAEnet

from torchvision.datasets import ImageFolder
import torchvision

# loda data
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = 'Shuguang_Village'
# Shuguang_Village  Sardinia  Gloucester  Texas

cae_root_path = '\data1\lcq\STCD\cae_result' # Path for storing generated images
split_root_path = '\data1\lcq\STCD\split_result' # path for storing subimages

NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

if dataset == 'Gloucester':
    IMAGE_SIZE = 128
elif dataset == 'Texas':
    IMAGE_SIZE = 64
else:
    IMAGE_SIZE = 32

if dataset == 'River' or dataset == 'Gloucester_' or dataset == 'California':
    #pre-event:optical   post-event:sar
    content_Folder = ImageFolder(os.path.join(split_root_path, dataset,
                                              'trainA'),
                                 transform=transform)
    style_Folder = ImageFolder(os.path.join(split_root_path, dataset,
                                            'trainB'),
                               transform=transform)
else:
    content_Folder = ImageFolder(os.path.join(split_root_path, dataset,
                                              'trainB'),
                                 transform=transform)
    style_Folder = ImageFolder(os.path.join(split_root_path, dataset,
                                            'trainA'),
                               transform=transform)

num_training_samples = len(content_Folder.samples)


# Custom sampler
class SamplerDef(object):
    def __init__(self, data_source, amount, generator=None):
        self.data_source = data_source
        self.amount = amount
        self.generator = generator

    def __iter__(self):
        n = self.amount
        return iter(torch.randperm(n, generator=self.generator).tolist())

    def __len__(self):
        return len(self.data_source)


mySampler = SamplerDef(data_source=content_Folder, amount=num_training_samples)
content_loader = DataLoader(content_Folder,
                            batch_size=BATCH_SIZE,
                            sampler=mySampler,
                            shuffle=False)
style_loader = DataLoader(style_Folder,
                          batch_size=BATCH_SIZE,
                          sampler=mySampler,
                          shuffle=False)

# Experimental environment
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def make_dir(image_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, IMAGE_SIZE, IMAGE_SIZE)
    save_image(img, name)


net = CDCAEnet()

#print(net)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


def train(net, content_loader, style_loader, NUM_EPOCHS):
    make_dir(os.path.join(cae_root_path, dataset, 'train_img'))
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        running_loss = 0.0
        for content, style in zip(content_loader, style_loader):
            content_img, _ = content  # no need for the labels
            content_img = content_img.to(device)
            style_img, _ = style  # no need for the labels
            style_img = style_img.to(device)

            optimizer.zero_grad()
            outputs, loss_ = net(content_img, style_img)
            loss_.backward()

            optimizer.step()
            running_loss += loss_.item()

        loss = running_loss / len(content_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}, Time: {:.2f} sec'.format(
            epoch + 1, NUM_EPOCHS, loss,
            time.time() - start_time))
        if epoch % 5 == 0:
            save_decoded_image(content_img.cpu().data,
                               name=os.path.join(cae_root_path, dataset,
                                                 'train_img',
                                                 f'original{epoch}.png'))
            save_decoded_image(outputs.cpu().data,
                               name=os.path.join(cae_root_path, dataset,
                                                 'train_img',
                                                 f'decoded{epoch}.png'))

        torch.save(obj=net.state_dict(),
                   f=os.path.join(cae_root_path, dataset, 'cdcae.pth'))

    return train_loss


device = get_device()
net.to(device)
#make_dir()
train_loss = train(net, content_loader, style_loader, NUM_EPOCHS)

plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(os.path.join(cae_root_path, dataset, 'Train_loss.png'))
