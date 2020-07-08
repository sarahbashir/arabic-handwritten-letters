import argparse

# process command line args
parser = argparse.ArgumentParser()

parser_model = parser.add_argument_group('model options')
parser_model.add_argument('--connections',choices=['plain','resnet'],default='resnet')
parser_model.add_argument('--size',type=int,default=20)


parser_opt = parser.add_argument_group('optimization options')
parser_opt.add_argument('--batch_size',type=int,default=16)
parser_opt.add_argument('--learning_rate',type=float,default=0.01)
parser_opt.add_argument('--epochs',type=int,default=10)
parser_opt.add_argument('--warm_start',type=str,default=None)

parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--dataset',choices=['mnist','cifar10','arabic_letters'])

parser_debug = parser.add_argument_group('debug options')
parser_debug.add_argument('--show_image',action='store_true')
parser_debug.add_argument('--print_delay',type=int,default=60)
parser_debug.add_argument('--log_dir',type=str)
parser_debug.add_argument('--eval',action='store_true')

args = parser.parse_args()

# load libraries
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# load data
if args.dataset=='cifar10':
    image_shape=[3,32,32]

    transform = transforms.Compose(
        [ transforms.RandomHorizontalFlip()
        , transforms.ToTensor()
        , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
        )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

if args.dataset=='mnist':
    image_shape=[1,28,28]

    transform = transforms.Compose(
        [ transforms.RandomHorizontalFlip()
        , transforms.ToTensor()
        , transforms.Normalize((0.5,), (0.5,))
        ])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
        )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

if args.dataset == 'arabic_letters':
    image_shape=[3,32,32]

    transform = transforms.Compose(
        [ transforms.RandomHorizontalFlip()
        , transforms.ToTensor()
        , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
        ])

    trainset = torchvision.datasets.ImageFolder(
        root='/opt/anaconda3/pkgs/torchvision-0.5.0-py37_cpu/lib/python3.7/site-packages/torchvision/datasets/ImageFolder/root/train',
        transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

    testset = torchvision.datasets.ImageFolder(
        root='/opt/anaconda3/pkgs/torchvision-0.5.0-py37_cpu/lib/python3.7/site-packages/torchvision/datasets/ImageFolder/root/test_run',
        transform=transform
        )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

# show image
if args.show_image:
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

# define the model
def conv3x3(channels_in, channels_out, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        channels_in,
        channels_out,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=1,
        bias=False
        )

class ResnetBlock(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            stride=1,
            downsample=None,
            use_bn = True
            ):
        super(ResnetBlock, self).__init__()
        norm_layer = torch.nn.BatchNorm2d
        self.use_bn = use_bn
        #use torch.view to reshape 3rd order to 1st order
        self.conv1 = conv3x3(channels_in, channels_out, stride)
        if self.use_bn:
            self.bn1 = norm_layer(channels_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels_out, channels_out)
        if self.use_bn:
            self.bn2 = norm_layer(channels_out)
        self.downsample = downsample
        self.stride = stride

    #directy connect conv layer to linear layer or global avg pooling
    #conv2D to linear
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        #change shape of identity
        #identity is 32x32x16 (i,j,k)
        #out is 32x32x32
        #w is 16x32 (k,l)
        #w = torch.tensor(16,32, requires_grad=True)
        #identity = torch.einsum('ijk,kl -> ijl', identity, w)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

import functools
image_size = functools.reduce(lambda x, y: x * y, image_shape, 1)

class Net(nn.Module):
    def __init__(self, block, layers):
        super(Net, self).__init__()
        self.conv = conv3x3(3,16)
        self.bn = nn.BatchNorm2d(16)
        self.channels_in = 16
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.layers(block, 16, layers[0])
        self.layer2 = self.layers(block, 32, layers[0], 2)
        self.layer3 = self.layers(block, 64, layers[1], 2)

        self.avg_pool = nn.AvgPool2d(3)
        #make the blocks here

        #one output for each class
        self.fc = nn.Linear(256,28)


    def layers(self, block, channels_out, blocks, stride=1):
        downsample = None
        if (stride != 1 or self.channels_in != channels_out):
            downsample = nn.Sequential(
            nn.Conv2d(self.channels_in, channels_out, kernel_size = 1, stride = stride)
            ,
            nn.BatchNorm2d(channels_out))

        layers = []
        layers.append(block(self.channels_in, channels_out, stride, downsample))
        self.channels_in = channels_out
        for i in range (1,blocks):
            layers.append(block(channels_out, channels_out))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



        #out = self.RelLu(self.bn1(selv.conv1(out)))
        #out - self.avg_pool2d(out, out.size()[3])
        #out = x.view(args.batch_size,image_size)


net = Net(ResnetBlock,[3,3,3])


# load pretrained model
# if args.warm_start is not None:
#     print('warm starting model from',args.warm_start)
#     model_dict = torch.load(os.path.join(args.warm_start,'model'))
#     net.load_state_dict(model_dict['model_state_dict'])

# create save dir
# log_dir = args.log_dir
# if log_dir is None:
#     log_dir = 'log/'+str(datetime.datetime.now())
#
# try:
#     os.mkdir(log_dir)
# except FileExistsError:
#     print('cannot create log dir,',log_dir,'already exists')
#     sys.exit(1)
#
# writer = SummaryWriter(log_dir=log_dir)

# train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
net.train()

total_iter = 0
last_print = 0

steps = 0
for epoch in range(args.epochs):
    for i, data in enumerate(trainloader):
        steps += 1
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # accuracy
        prediction = torch.argmax(outputs,dim=1)
        accuracy = (prediction==labels).float().mean()

        # tensorboard
        # writer.add_scalar('train/loss', loss.item(), steps)
        # writer.add_scalar('train/accuracy', accuracy.item(), steps)

        # print statistics
        total_iter += 1
        if time.time() - last_print > args.print_delay:
            print(datetime.datetime.now(),'epoch = ',epoch,'steps=',steps,'batch/sec=',total_iter/args.print_delay)
            total_iter = 0
            last_print = time.time()

    # torch.save({
    #         'epoch':epoch,
    #         'model_state_dict': net.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss':loss
    #     }, os.path.join(log_dir,'model'))


# test set
if args.eval:
    print('evaluating model')
    net.eval()

    loss_total = 0
    accuracy_total = 0
    for i, data in enumerate(testloader):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # accuracy
        prediction = torch.argmax(outputs,dim=1)
        accuracy = (prediction==labels).float().mean()

        # update variables
        loss_total += loss.item()
        accuracy_total += accuracy.item()

    print('loss=',loss_total/i)
    print('accuracy=',accuracy_total/i)
