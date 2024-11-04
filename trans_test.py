import os
import time
import torch
import torchvision
import torch.nn as nn
import tqdm as tqdm
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from net import CDCAEnet
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision

############################################################## test

dataset = 'Shuguang_Village'
# Shuguang_Village  Sardinia  Gloucester  Texas

# cae_root_path = 'cd\evidence_cd\cae_result' # old
# split_root_path = 'cd\evidence_cd\split_result' # old

cae_root_path = '\data1\lcq\STCD\cae_result' # Path for storing generated images
split_root_path = '\data1\lcq\STCD\split_result' # path for storing subimages
if dataset == 'Gloucester':
    IMAGE_SIZE = 128
elif dataset == 'Texas':
    IMAGE_SIZE = 64
else:
    IMAGE_SIZE = 32

style_batch_size = 512

##loda data
transform = transforms.Compose([
    #   transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if dataset == 'River' or dataset == 'Gloucester_' or dataset == 'California':
    #事前光   事后sar
    content_Folder = ImageFolder(os.path.join(split_root_path, dataset,
                                              'testA'),
                                 transform=transform)
    style_Folder = ImageFolder(os.path.join(split_root_path, dataset, 'testB'),
                               transform=transform)
else:
    content_Folder = ImageFolder(os.path.join(split_root_path, dataset,
                                              'testB'),
                                 transform=transform)
    style_Folder = ImageFolder(os.path.join(split_root_path, dataset, 'testA'),
                               transform=transform)
#num_training_samples = len(content_Folder.samples)

content_loader = DataLoader(content_Folder, batch_size=1, shuffle=False)
style_loader = DataLoader(style_Folder,
                          batch_size=style_batch_size,
                          shuffle=True)


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
    img_num = img.shape[0]
    img = torch.sum(img, dim=0) / img_num

    save_image(img, name)


def style_transfer(net, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    g_t, _ = net(content, style)
    return g_t


net = CDCAEnet()
net.eval()
device = get_device()
net.load_state_dict(
    torch.load(os.path.join(cae_root_path, dataset, 'cdcae.pth'),
               map_location=device))
net.to(device)


def test(net, content_loader, style_loader):
    make_dir(os.path.join(cae_root_path, dataset, 'test_img'))
    num = 0
    with tqdm(total=len(content_loader)) as t:
        for step, content in enumerate(content_loader):
            output = torch.zeros(len(content_loader), 3, IMAGE_SIZE,
                                 IMAGE_SIZE)
            for n, style in enumerate(style_loader):
                style_img, _ = style  # no need for the labels
                content_img, _ = content  # no need for the labels
                content_img = content_img.expand_as(style_img)
                style_img = style_img.to(device)
                content_img = content_img.to(device)

                output[n * style_batch_size:n * style_batch_size +
                       content_img.shape[0]] = style_transfer(
                           net, content_img, style_img)

            t.set_description(desc="test")
            t.set_postfix(steps=step)
            t.update(1)

            save_decoded_image(output.cpu().data,
                               name=os.path.join(
                                   cae_root_path, dataset, 'test_img',
                                   'Image_style_{:03d}.png'.format(num)))

            num += 1


test(net, content_loader, style_loader)
