#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   DataLoader.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/3/16 16:28
------------      --------    -----------

"""
from torchvision import datasets, transforms
import torch

class FashionMNISTLoader:

    def __init__(self,device_config):
        self.dataset_name = ""
        if device_config == 2 or device_config == 1: # CPU parallel
            self.num_workers = 4
        else:
            self.num_workers=0
        self.dataloader = self.load_FashionMNIST_data(num_workers=self.num_workers)

    def load_FashionMNIST_data(self, batch_size=128, num_workers=8):
        print("Loading Data......")
        """Load dataset"""
        batchsz = batch_size
        self.dataset_name = "FashionMNIST"
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),

            # transforms.RandomVerticalFlip(p=0.1),
            # transforms.RandomRotation(degrees=(-30, 30)),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),  # converting images to tensor
            transforms.Normalize(mean=(0.5), std=(0.5))
            # if the image dataset is black and white image, there can be just one number.
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])

        train_data = datasets.FashionMNIST(root='../data', train=True, transform=transform_train, download=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsz, shuffle=True,
                                                   num_workers=num_workers)

        val_dataset = datasets.FashionMNIST('../data',
                                            train=False,
                                            transform=transform_train, download=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batchsz,
                                                 shuffle=False,
                                                 num_workers=num_workers)

        test_data = datasets.FashionMNIST(root='../data', train=False, transform=transform_test, download=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsz, shuffle=True)

        return train_loader, val_loader, test_loader

class MNISTLoader:

    def __init__(self, device_config):
        self.dataset_name = ""
        if device_config == 2 or device_config == 1: # CPU parallel
            self.num_workers = 4
        else:
            self.num_workers=0
        self.dataloader = self.load_MNIST_data(num_workers=self.num_workers)

    def load_MNIST_data(self, batch_size=128, num_workers=8):
        print("Loading Data......")
        self.dataset_name = "MNIST"
        """Load MNIST data set"""
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),

            # transforms.RandomVerticalFlip(p=0.1),
            # transforms.RandomRotation(degrees=(-30, 30)),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),  # converting images to tensor
            transforms.Normalize(mean=(0.5), std=(0.5))
            # if the image dataset is black and white image, there can be just one number.
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
        train_data = datasets.MNIST(root='../data/',
                                    train=True,
                                    transform=transform_train,
                                    download=True)

        test_data = datasets.MNIST(root='../data/',
                                   train=False,
                                   transform=transform_test)

        val_dataset = datasets.MNIST('../data',
                                     train=False,
                                     transform=transform_train, download=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers)

        # return a iteritor
        # shuffle：change the order or not
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)
        return train_loader, val_loader, test_loader

class CIFARLoader:
    def __init__(self, device_config):
        self.dataset_name = ""
        if device_config == 1 or device_config == 2:  # CPU parallel
            self.num_workers = 6
            self.pin_memory = True
        else:
            self.num_workers = 0
            self.pin_memory = False
        self.dataloader = self.load_CIFAR10_data(num_workers=self.num_workers)

    def load_CIFAR10_data(self, batch_size=128, num_workers=8):
        print("Loading Data......")
        """Load dataset"""
        self.dataset_name = "CIFAR"
        batchsz = batch_size
        transform_train = transforms.Compose([
            # transforms.Resize((64, 64)),
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),  # converting images to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            # if the image dataset is black and white image, there can be just one number.
        ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
            # transforms.CenterCrop(64),  # 依据给定的size从中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        # Download the dataset
        # train_data = datasets.CIFAR10(root='./cifar', train=True, transform=transforms.Compose([
        #     transforms.Resize((64, 64)),
        #     transforms.ToTensor()
        # ]), download=True)
        train_data = datasets.CIFAR10(root='../data/CIFAR10', train=True, transform=transform_train, download=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsz, shuffle=True,
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory)

        val_dataset = datasets.CIFAR10('../data/CIFAR10',
                                       train=True,
                                       transform=transform_test, download=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batchsz,
                                                 shuffle=False,
                                                 num_workers=num_workers)

        # test_data = datasets.CIFAR10(root='./cifar', train=False, transform=transforms.Compose([
        #     transforms.Resize((64, 64)),
        #     transforms.ToTensor()
        # ]), download=True)
        test_data = datasets.CIFAR10(root='../data/CIFAR10', train=False, transform=transform_test, download=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsz, shuffle=True, num_workers=num_workers)

        return train_loader, val_loader, test_loader

class CIFARLoaderWithoutPreprocess:
    def __init__(self):
        self.dataset_name = ""
        self.dataloader = self.load_CIFAR10_data_without_preprocess()

    def load_CIFAR10_data_without_preprocess(self, batch_size=128, num_workers=8):
        print("Loading Data......")
        """Load dataset"""
        self.dataset_name = "CIFAR"
        batchsz = batch_size
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),  # converting images to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            # if the image dataset is black and white image, there can be just one number.
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
            # transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        # Download the dataset
        # train_data = datasets.CIFAR10(root='./cifar', train=True, transform=transforms.Compose([
        #     transforms.Resize((64, 64)),
        #     transforms.ToTensor()
        # ]), download=True)
        train_data = datasets.CIFAR10(root='../data/CIFAR10', train=True, transform=transform_train, download=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsz, shuffle=True,
                                                   num_workers=num_workers)

        val_dataset = datasets.CIFAR10('../data/CIFAR10',
                                       train=True,
                                       transform=transform_test, download=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batchsz,
                                                 shuffle=False,
                                                 num_workers=num_workers)

        # test_data = datasets.CIFAR10(root='./cifar', train=False, transform=transforms.Compose([
        #     transforms.Resize((64, 64)),
        #     transforms.ToTensor()
        # ]), download=True)
        test_data = datasets.CIFAR10(root='../data/CIFAR10', train=False, transform=transform_test, download=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsz, shuffle=True, num_workers=num_workers)

        return train_loader, val_loader, test_loader