#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Driver.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/3/18 22:20
------------      --------    -----------

"""
import TrainProcess
import DataLoader
import Net
import multiprocessing as mp
class Driver:
    def __init__(self, dataset):
        if dataset == "MNIST":
            self.MNIST_dirver()
        elif dataset == "FashionMNIST":
            self.FashionMNIST_dirver()
        elif dataset == "CIFAR":
            self.CIFAR_dirver()


    def MNIST_dirver(self):
        for device_config in range(0,5):
            print("start train on MNIST by Lenet on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.MNISTLoader(device_config=device_config),
                     net=Net.LeNet(input_channel=1,H=32,W=32),
                     device_config=device_config)
        for device_config in range(0,5):
            print("start train on MNIST by Modified Lenet on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.MNISTLoader(device_config=device_config),
                     net=Net.ModifiedLeNet(input_channel=1,H=32,W=32),
                     device_config=device_config)

        for device_config in range(0,5):
            print("start train on MNIST by MobileNetV3 on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.MNISTLoader(device_config=device_config),
                     net=Net.mobilenet_v3_large(input_channel=1),
                     device_config=device_config)
        #
        # for device_config in range(5):
        #     TrainProcess.TrainProcess(dataLoader=DataLoader.FashionMNISTLoader(),
        #                               net=Net.swin_tiny_patch4_window7_224(in_channel=1, num_classes=TrainProcess.Config.num_classes),
        #                               device_config=device_config)

    def FashionMNIST_dirver(self):
        for device_config in range(0,5):
            print("start train on FashionMNIST by Lenet on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.FashionMNISTLoader(device_config=device_config),
                                      net=Net.LeNet(input_channel=1,H=32,W=32),
                                      device_config=device_config)

        for device_config in range(0,5):
            print("start train on FashionMNIST by Modified Lenet on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.FashionMNISTLoader(device_config=device_config),
                                      net=Net.ModifiedLeNet(input_channel=1, H=32, W=32),
                                      device_config=device_config)

        for device_config in range(0,5):
            print("start train on FashionMNIST by MobileNetV3 on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.FashionMNISTLoader(device_config=device_config),
                     net=Net.mobilenet_v3_large(input_channel=1),
                     device_config=device_config)
        # for device_config in range(5):
        #     TrainProcess.TrainProcess(dataLoader=DataLoader.FashionMNISTLoader(),
        #                               net=Net.swin_tiny_patch4_window7_224(in_channel=1, num_classes=TrainProcess.Config.num_classes),
        #                               device_config=device_config)

    def CIFAR_dirver(self):
        for device_config in range(0,5):
            print("start train on CIFAR by Lenet on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=device_config),
                     net=Net.LeNet(input_channel=3, H=64, W=64),
                     device_config=device_config)

        for device_config in range(0,5):
            print("start train on CIFAR by Modified Lenet on device config:",device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=device_config),
                     net=Net.ModifiedLeNet(input_channel=3, H=64, W=64),
                     device_config=device_config)

        for device_config in range(0,5):
            print("start train on CIFAR by MobileNetV3 on device config:", device_config)
            TrainProcess.TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=device_config),
                     net=Net.mobilenet_v3_large(input_channel=3),
                     device_config=device_config)
        #
        # for device_config in range(5):
        #     TrainProcess.TrainProcess(dataLoader=DataLoader.FashionMNISTLoader(),
        #                               net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=TrainProcess.Config.num_classes),
        #                               device_config=device_config)



if __name__ == "__main__":
    # mp.freeze_support()

    Driver(dataset="CIFAR")
    Driver(dataset="MNIST")
    Driver(dataset="FashionMNIST")


        # print(device_config)