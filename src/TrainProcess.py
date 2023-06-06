#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   TrainProcess.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/3/16 16:54
------------      --------    -----------

"""
from datetime import datetime

import numpy as np
import multiprocessing as mp
import torch
from matplotlib import pyplot as plt, animation
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm

from AnimationDraw import AnimationDraw
import DataLoader
import Net

# VIDEO_DIR = "../out_eval_32CIfar/videos/"
# IMAGE_DIR = "../out_eval_32CIfar/images/"
# LOG_PATH = "../out_eval_32CIfar/log/"
# GIF_DIR = "../out_eval_32CIfar/gif/"

# VIDEO_DIR = "../out/NoPreprocess/32-RandomCrop/"
# IMAGE_DIR = "../out/NoPreprocess/32-RandomCrop/"
# LOG_PATH = "../out/NoPreprocess/32-RandomCrop/"
# GIF_DIR = "../out/NoPreprocess/32-RandomCrop/"
# MODEL_DIR = "../out/NoPreprocess/32-RandomCrop/"

#
MODEL_DIR = "../out/cpu_parallel/models/"

VIDEO_DIR = "../out/cpu_parallel/videos/"
IMAGE_DIR = "../out/cpu_parallel/images/"
LOG_PATH = "../out/cpu_parallel/log/"
GIF_DIR = "../out/cpu_parallel/gif/"



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("cuda aviailabe", torch.cuda.is_available())

class Config:

    patch_size = 4
    num_classes = 10
    batch_size = 128
    epoch = 30
    val_epoch = 1
    momentum = 0.15
    # alpha = 1e-3
    learn_rate = 0.001
    weight_decay = 0.01
    num_workers = 8

    plt_time_loc_x = 60
    plt_time_loc_y = 1.5

    print_per_step = 200
    plt_per_step = 200
    print_per_step_test = 1000
    plt_per_seccond = 16

class train_info:

    def __init__(self):
        super(train_info, self).__init__()
        self.time_record_list = mp.Manager().list()
        self.loss_list_train = mp.Manager().list()
        self.loss_list_line = mp.Manager().list()
        self.accuracy_list_train = mp.Manager().list()
        self.accuracy_list_line = mp.Manager().list()
        # self.artists = mp.Manager().list()


class TrainProcess:

    def __init__(self, dataLoader, net, device_config, cpu_parallel_process:int = 3):
        """

        Args:
            dataLoader:
            net: LeNet / ModifiedLenet / MobileNetV3
            device_config:
                    device_config == 0: train on cpu
                    device_config == 1: train on cpu parallely
                    device_config == 2: train on gpu and cpu
                    device_config == 3: train on gpu
                    device_config == 4: train on gpu with data parallel
        """

        self.net_name = net.net_name
        self.dataset_name = dataLoader.dataset_name
        self.cpu_parallel_process = cpu_parallel_process

        # set dataset
        if self.dataset_name == "CIFAR":
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        elif self.dataset_name == "FashionMNIST":
            self.classes = (
            'T-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
        elif self.dataset_name == "MNIST":
            self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        # set config
        if device_config == 0:
            self.train, self.val, self.test = dataLoader.dataloader
            self.net = net.to("cpu")
            self.criterion = nn.CrossEntropyLoss()
            # self.optimizer = optim.Adam(self.net.parameters(), lr=Config.learn_rate)
            self.optimizer = optim.SGD(self.net.parameters(), lr=Config.learn_rate)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
            self.device_config = "cpu"
            self.train_step()
        elif device_config == 1:
            self.train, self.val, self.test = dataLoader.dataloader
            self.device_config = "cpu_parallel"
            self.net = net.to("cpu")
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.net.parameters(), lr=Config.learn_rate)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
            self.train_on_cpu_parallel(cpu_parallel_process)

        elif device_config == 2:
            self.train, self.val, self.test = dataLoader.dataloader
            self.net = nn.DataParallel(net).to(device)
            self.criterion = nn.CrossEntropyLoss().to(device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=Config.learn_rate)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
            self.device_config = "gpu_and_cpu"
            self.train_step()
        elif device_config == 3:
            self.train, self.val, self.test = dataLoader.dataloader
            self.net = net.to(device)
            self.criterion = nn.CrossEntropyLoss().to(device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=Config.learn_rate)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
            self.device_config = "gpu"
            self.train_step()
        else:
            self.train, self.val, self.test = dataLoader.dataloader
            self.net = nn.DataParallel(net).to(device)
            self.criterion = nn.CrossEntropyLoss().to(device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=Config.learn_rate)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
            self.device_config = "gpu_data_parallel"
            self.train_step()
        if self.device_config == "cpu_parallel":
            torch.save(self.net, MODEL_DIR+self.dataset_name + "_" +self.device_config + "_" + self.net_name + ".pth")
        else:
            torch.save(self.net, MODEL_DIR+self.dataset_name + "_" +self.device_config + "_" + self.net_name+"_cpu_"+ str(self.cpu_parallel_process)+".pth")

        self.net.eval()
        self.val_step(Config.val_epoch)
        self.test_step()


    def train_on_sub_cpu_process(self, epoch, cur_process, steps, start_time, last_end_time,log_file_path,
                                 train_img_info):
        log_file = open(log_file_path, "a")
        for batch_idx, (data, label) in tqdm(enumerate(self.train), total=len(self.train),
                                             desc='Epoch {} Process:{}'.format(epoch, cur_process)):

            self.optimizer.zero_grad()
            outputs = self.net(data)
            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()  # update the arguments by grad
            end_time = datetime.now()
            last_time_diff = (end_time - last_end_time).seconds
            # print results ever 100 times
            # print("time:", end_time," time diff:",time_diff % Config.plt_per_seccond)
            if steps % Config.print_per_step == 0:

                _, predicted = torch.max(outputs, 1)
                correct = int(sum(predicted == label))
                accuracy = correct / Config.batch_size  # calculate the accuracy
                time_diff = (end_time - start_time).seconds
                time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                msg = "Epoch {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                train_img_info.loss_list_train.append(loss.item())
                train_img_info.accuracy_list_train.append(accuracy)
                log_file.write(msg.format(epoch, loss, accuracy, time_usage) + "\n")
                print(msg.format(epoch, loss, accuracy, time_usage))
                # if steps % Config.plt_per_step == 0:
                if last_time_diff > Config.plt_per_seccond or steps == 0:
                    # print("time:", end_time, " time diff:", last_time_diff % Config.plt_per_seccond)
                    last_end_time = end_time
                    train_img_info.accuracy_list_line.append(accuracy)
                    train_img_info.time_record_list.append(str(time_usage))
                    train_img_info.loss_list_line.append(loss.item())
                    plt.pause(0.1)
            # self.scheduler.step()  # update learning rate

            steps += 1
        log_file.close()
        # return


    def train_on_cpu_parallel(self,  num_process):
        log_file_path = LOG_PATH + self.dataset_name + "_" + self.device_config + "_" + self.net_name + "_cpu_"+ str(self.cpu_parallel_process) +".txt"
        log_file = open(log_file_path, "w")
        img_path = IMAGE_DIR + self.dataset_name + "_" + self.device_config + "_" + self.net_name + "_cpu_"+ str(self.cpu_parallel_process)
        steps = 0

        train_img_info = train_info()
        fig, ax = plt.subplots()
        # loss_list_train = []
        # loss_list_line = []
        # accuracy_list_train = []
        # accuracy_list_line = []
        artists = []


        line_loss = ax.plot([i for i in range(len(train_img_info.loss_list_train))], train_img_info.loss_list_train, label='loss',
                            c='blue', ls='-')
        line_acc = ax.plot([i for i in range(len(train_img_info.accuracy_list_train))], train_img_info.accuracy_list_train, label='accuracy', c='r',
                           ls='-')
        epoch_in_each_process = Config.epoch // num_process
        remainder_epoch = Config.epoch % num_process
        # if cur_process < remainder_epoch:
        #     epoch_in_each_process += 1
        start_time = datetime.now()
        last_end_time = start_time
        log_file.close()

        print("Training & Evaluating......")
        for epoch in range(epoch_in_each_process):

            print("Epoch {:3}".format(epoch + 1))
            model = self.net
            num_processes = num_process  # 设置4个进程
            # NOTE: this is required for the ``fork`` method to work
            model.share_memory()

            processes = []
            for rank in range(num_processes):
                # 4 个进程，每个进程epoch为150，也就是说其实迭代了 4*150 = 600 次 !!!
                process = mp.Process(target=self.train_on_sub_cpu_process, args=(epoch, rank, steps, start_time, last_end_time,log_file_path,
                                 train_img_info))
                process.start()
                processes.append(process)
            for process in processes:
                process.join()

        for epoch in range(remainder_epoch):
            self.train_on_sub_cpu_process(epoch+epoch_in_each_process, epoch,  steps, start_time, last_end_time,log_file_path,
                                 train_img_info)

        # add to drawing
        Config.plt_time_loc_x = len(train_img_info.accuracy_list_line)-2
        print("plt:",Config.plt_time_loc_x)
        for sub_item in range(len(train_img_info.accuracy_list_line)):
            line_loss = ax.plot([i for i in range(sub_item)], [train_img_info.loss_list_line[i_item] for i_item in range(sub_item)],c='blue', ls='-')
            line_acc = ax.plot([i for i in range(sub_item)], [train_img_info.accuracy_list_line[i_item] for i_item in range(sub_item)], c='r', ls='-')
            show_time = [
                ax.text(Config.plt_time_loc_x, Config.plt_time_loc_y, s="train time:" + train_img_info.time_record_list[sub_item], ha='right')]
            artists.append([line_acc, line_loss, show_time])
        log_file = open(log_file_path, "a")
        train_end_time = datetime.now()
        train_time_diff = (train_end_time - start_time).seconds
        train_time_usage = '{:3}m{:3}s'.format(int(train_time_diff / 60), train_time_diff % 60)
        msg = " Time usage of train:{:9}."
        log_file.write(msg.format(train_time_usage) + "\n")

        ax.legend(loc='upper right')
        ax.set_title(
            "In " + str(self.dataset_name) + " " + str(self.device_config) + " " + str(self.net_name) + " epoch:" + str(
                Config.epoch)+" cpu:"+ str(self.cpu_parallel_process))
        self.draw_train_mp4(fig, artists)
        self.draw_train_gif(fig, artists)

        """
        Draw the diagram of accuracy and loss in train
        """
        plt.title(self.dataset_name + " " + self.device_config + " in " + self.net_name)
        plt.ylim(0, 1)

        #
        # plt.savefig(img_path + "_loss.png")
        # plt.show()

        plt.plot([i for i in range(len(train_img_info.accuracy_list_train))], train_img_info.accuracy_list_train, c='red', label="accuracy")
        plt.plot([i for i in range(len(train_img_info.loss_list_train))], train_img_info.loss_list_train, c="blue", label="loss")
        plt.legend(loc='upper right')
        plt.savefig(img_path + "_loss_accuracy.png")
        plt.show()

        plt.plot([i for i in range(len(train_img_info.loss_list_train))], train_img_info.loss_list_train, c="blue", label="loss")
        plt.title(self.dataset_name + " " + self.device_config + " in " + self.net_name)

        plt.legend(loc='upper right')
        plt.savefig(img_path + "_loss.png")
        plt.show()

        plt.plot([i for i in range(len(train_img_info.accuracy_list_train))], train_img_info.accuracy_list_train,
                 c='red', label="accuracy")
        plt.title(self.dataset_name + " " + self.device_config + " in " + self.net_name)
        plt.ylim(0, 1)
        plt.legend(loc='upper right')
        plt.savefig(img_path + "_accuracy.png")
        plt.show()
        log_file.close()





    def train_step(self):
        """
        train excepts the cpu parallel
        Returns:

        """
        log_file = open(LOG_PATH + self.dataset_name + "_" +self.device_config + "_" + self.net_name + ".txt", "w")
        img_path = IMAGE_DIR + self.dataset_name +"_" + self.device_config + "_" + self.net_name +"_"
        steps = 0 # record the iterate times

        loss_list_train = []
        loss_list_line = []
        accuracy_list_train = []
        accuracy_list_line = []
        artists = [] # use in plt.animation
        time_record_list = []

        fig, ax = plt.subplots()
        line_loss = ax.plot([i for i in range(len(loss_list_train))], loss_list_train, label='loss',
                            c='blue', ls='-')
        line_acc = ax.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train, label='accuracy', c='r',
                           ls='-')
        ax.set_title("In "+str(self.dataset_name)+" "+str(self.device_config)+" "+str(self.net_name)+" epoch:"+str(Config.epoch))
        start_time = datetime.now()
        last_end_time = start_time

        print("Training & Evaluating......")
        for epoch in range(Config.epoch):

            print("Epoch {:3}".format(epoch + 1))
            for batch_idx, (data, label) in tqdm(enumerate(self.train), total=len(self.train),
                                                 desc='Epoch {} '.format(epoch)):
            # for data, label in self.train:
                # cpu --> gpu
                if self.device_config[0] == "g" and self.device_config != "gpu_and_cpu":
                    data, label = data.to(device), label.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.net(data).to(device)
                    loss = self.criterion(outputs, label).to(device)
                    loss.backward()
                    self.optimizer.step()  # update the arguments by grad
                # cpu-->gpu
                elif self.device_config == "gpu_and_cpu":
                    data, label = data.to(device), label.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.net(data).to(device)
                    loss = self.criterion(outputs, label).to(device)
                    loss.backward()
                    self.optimizer.step()  # update the arguments by grad
                else:
                    self.optimizer.zero_grad()
                    outputs = self.net(data)
                    loss = self.criterion(outputs, label)
                    loss.backward()
                    self.optimizer.step()  # update the arguments by grad
                end_time = datetime.now()
                last_time_diff = (end_time - last_end_time).seconds
                # print results ever 100 times
                # print("time:", end_time," time diff:",time_diff % Config.plt_per_seccond)
                if steps % Config.print_per_step == 0:

                    _, predicted = torch.max(outputs, 1)
                    correct = int(sum(predicted == label))
                    accuracy = correct / Config.batch_size  # calculate the accuracy
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Epoch {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    loss_list_train.append(loss.item())
                    accuracy_list_train.append(accuracy)
                    log_file.write(msg.format(epoch, loss, accuracy, time_usage)+"\n")
                    print(msg.format(epoch, loss, accuracy, time_usage))
                # if steps % Config.plt_per_step == 0:
                    if last_time_diff > Config.plt_per_seccond or steps == 0:
                       #  print("time:", end_time, " time diff:", last_time_diff % Config.plt_per_seccond)
                        last_end_time = end_time
                        accuracy_list_line.append(accuracy)
                        time_record_list.append(str(time_usage))
                        loss_list_line.append(loss.item())
                self.scheduler.step()  # update learning rate
                steps += 1

        # the last training
        _, predicted = torch.max(outputs, 1)
        correct = int(sum(predicted == label))
        accuracy = correct / Config.batch_size  # calculate the accuracy
        time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
        msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
        log_file.write(msg.format(steps, loss, accuracy, time_usage) + "\n")
        msg = " Time usage of train:{:9}."
        log_file.write(msg.format(time_usage)+"\n")

        # train_end_time = datetime.now()
        # train_time_diff = (train_end_time - start_time).seconds
        # time_usage = '{:3}m{:3}s'.format(int(train_time_diff / 60), time_diff % 60)
        # msg = " Time usage of train:{:9}."
        # log_file.write(msg.format(time_usage)+"\n")
        # add to drawing
        # if steps % Config.plt_per_step == 0:
        loss_list_train.append(loss.item())
        accuracy_list_train.append(accuracy)
        accuracy_list_line.append(accuracy)
        time_record_list.append(str(time_usage))
        loss_list_line.append(loss.item())
        Config.plt_time_loc_x = len(accuracy_list_line)-2
        print("plt:",Config.plt_time_loc_x)
        for sub_item in range(len(accuracy_list_line)):
            line_loss = ax.plot([i for i in range(sub_item)], [loss_list_line[i_item] for i_item in range(sub_item)],c='blue', ls='-')
            line_acc = ax.plot([i for i in range(sub_item)], [accuracy_list_line[i_item] for i_item in range(sub_item)], c='r', ls='-')
            show_time = [
                ax.text(Config.plt_time_loc_x, Config.plt_time_loc_y, s="train time:" + time_record_list[sub_item], ha='right')]
            artists.append([line_acc, line_loss, show_time])



        ax.legend(loc='upper right')
        self.draw_train_gif(fig, artists)
        self.draw_train_mp4(fig, artists)

        """
        Draw the diagram of accuracy and loss in train
        """
        # plt.show()

        plt.title(self.dataset_name + " " + self.device_config + " in " + self.net_name)
        plt.ylim(0, 1)

        #
        # plt.savefig(img_path + "_loss.png")
        # plt.show()
        plt.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train, c='red', label="accuracy")
        plt.plot([i for i in range(len(loss_list_train))], loss_list_train, c="blue", label="loss")
        plt.legend(loc='upper right')
        plt.savefig(img_path + "_loss_accuracy.png")
        plt.show()

        plt.plot([i for i in range(len(loss_list_train))], loss_list_train, c="blue", label="loss")
        plt.title(self.dataset_name + " " + self.device_config + " in " + self.net_name)

        plt.legend(loc='upper right')
        plt.savefig(img_path + "_loss.png")
        plt.show()

        plt.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train,
                 c='red', label="accuracy")
        plt.title(self.dataset_name + " " + self.device_config + " in " + self.net_name)
        plt.ylim(0, 1)
        plt.legend(loc='upper right')
        plt.savefig(img_path + "_accuracy.png")
        plt.show()
        log_file.close()






    def draw_train_mp4(self, fig, artists):
        plt.rcParams['animation.ffmpeg_path'] = 'E:/Software/ImageMagick/ffmpeg.exe'  # 修改为您的ffmpeg路径
        plt.rcParams['animation.writer'] = 'ffmpeg'
        # for a in artists:
        #     print("artist:",a)
        Writer = animation.FFMpegWriter  # 需安装ffmpeg
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani_acc_loss = AnimationDraw(fig, artists, interval=30)
        if self.device_config != "cpu_parallel":
            ani_acc_loss.save(VIDEO_DIR + self.dataset_name + "_" + self.device_config + "_" + self.net_name+"_"+"accuracy_loss.mp4", writer=writer)
        else:
            ani_acc_loss.save(VIDEO_DIR + self.dataset_name + "_" + self.device_config + "_" + self.net_name+"_cpu_" + str(self.cpu_parallel_process)+"accuracy_loss.mp4", writer=writer)

    def draw_train_gif(self, fig, artists):
        # plt.rcParams['animation.ffmpeg_path'] = 'E:/Software/ImageMagick/ffmpeg.exe'  # 修改为您的ffmpeg路径
        # plt.rcParams['animation.magick_path'] = 'E:/Software/ImageMagick/magick.exe'
        # plt.rcParams['animation.writer'] = 'imagemagick'
        # for a in artists:
        #     print("artist:",a)
        # Writer = animation.FFMpegWriter  # 需安装ffmpeg
        Writer = animation.ImageMagickWriter
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani_acc_loss = AnimationDraw(fig, artists, interval=30, repeat=True, repeat_delay=100)
        if self.device_config != "cpu_parallel":
            ani_acc_loss.save(GIF_DIR + self.dataset_name + "_" + self.device_config + "_" + self.net_name+"accuracy_loss.gif", dpi=80, writer=writer)
        else:
            ani_acc_loss.save(
                GIF_DIR + self.dataset_name + "_" + self.device_config + "_" + self.net_name + "_cpu_" + str(self.cpu_parallel_process) + "accuracy_loss.gif",
                writer=writer)

    '''
        Start Val
    '''

    def val_step(self, epoches):

        steps = 0
        val_loss = 0.
        val_correct = 0
        loss_list_val = []
        accuracy_list_val = []
        start_time = datetime.now()
        print("Validating......")
        log_file = open(LOG_PATH + self.dataset_name + "_" + self.device_config + "_" + self.net_name + ".txt", "a")

        #total_batch = len(val_dataset) // Config.batch_size
        for epoch in range(epoches):
            print("Epoch in Val {:3}".format(epoch + 1))
            for data, label in self.val:
                with torch.no_grad():
                    #  cpu --> gpu
                    if self.device_config[0] == "g":
                        data, label = data.to(device), label.to(device)
                    # cpu-->gpu
                    outputs = self.net(data)
                    #print("output:{},label{}".format(outputs,label))
                    loss = self.criterion(outputs, label)
                    val_loss += loss * Config.batch_size
                    _, predicted = torch.max(outputs, 1)
                    correct = int(sum(predicted == label))
                    val_correct += correct
                    accuracy = correct / Config.batch_size  # calculate the accuracy

                    if steps % Config.print_per_step == 0:

                        end_time = datetime.now()
                        time_diff = (end_time - start_time).seconds
                        time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                        msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                        print(msg.format(steps, loss, accuracy, time_usage))
                        log_file.write(msg.format(steps, loss, accuracy, time_usage)+"\n")
                        loss_list_val.append(loss.item())
                        accuracy = correct / Config.batch_size
                        accuracy_list_val.append(accuracy)

                    steps += 1
        log_file.close()
        """
        Draw the diagram of accuracy and loss in test
        """

    """
        Start test
    """
    def test_step(self):

        steps = 0
        test_loss = 0.
        test_correct = 0
        loss_list_test = []
        accuracy_list_test = []
        start_time = datetime.now()
        classes_correct_dict = {'plane': 0, 'car': 0, 'bird': 0, 'cat': 0, 'deer': 0,
                                'dog': 0, 'frog': 0, 'horse': 0, 'ship': 0, 'truck': 0}
        classes_all_num_dict = {'plane': 0, 'car': 0, 'bird': 0, 'cat': 0, 'deer': 0,
                                'dog': 0, 'frog': 0, 'horse': 0, 'ship': 0, 'truck': 0}
        if self.device_config == "cpu_parallel":
            log_file = open(LOG_PATH+self.dataset_name+"_"+self.device_config+"_"+self.net_name+"_cpu_"+str(self.cpu_parallel_process)+".txt", "a")
            img_path = IMAGE_DIR + self.dataset_name +  self.device_config + "_" + self.net_name+"_cpu_"+str(self.cpu_parallel_process)
        else:
            log_file = open(
                LOG_PATH + self.dataset_name + "_" + self.device_config + "_" + self.net_name + ".txt", "a")
            img_path = IMAGE_DIR + self.dataset_name + self.device_config + "_" + self.net_name
        print("Start testing......")
        for data, label in self.test:
            with torch.no_grad():
                #  cpu --> gpu
                if self.device_config[0] == "g":
                    data, label = data.to(device), label.to(device)
                # cpu-->gpu

                outputs = self.net(data)
                #print("output:{},label{}".format(outputs,label))
                loss = self.criterion(outputs, label)
                test_loss += loss * Config.batch_size
                _, predicted = torch.max(outputs, 1)
                correct = int(sum(predicted == label))
                test_correct += correct
                accuracy = correct / Config.batch_size  # calculate the accuracy
                classes_correct_dict, classes_all_num_dict = self.identify_classes_accuracy(classes_correct_dict,
                                                                                            classes_all_num_dict,
                                                                                            predicted, label)
                # print("test correct:{}, correct:{}, accuracy:{}".format(test_correct, correct, accuracy))

                if steps % Config.print_per_step_test == 0:


                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    print(msg.format(steps, loss, accuracy, time_usage))
                    log_file.write(msg.format(steps, loss, accuracy, time_usage)+"\n")
                loss_list_test.append(loss.item())
                accuracy_list_test.append(accuracy)
                steps += 1

        """
        Draw the diagram of accuracy and loss in test
        """
        train_times = np.linspace(1, len(loss_list_test), len(loss_list_test))
        plt.plot(train_times, loss_list_test)
        plt.title("Loss of GPU training processs in test")
        plt.show()
        plt.plot([i for i in range(len(accuracy_list_test))], accuracy_list_test)
        plt.title("Accuracy of GPU training process in test")
        plt.show()
        # show the last data and predicted
        for i in range(12):
            plt.subplot(3, 4, i + 1)  # subplot(m,n,i)
            plt.tight_layout()

            # plt.imshow(data.cpu().numpy()[i][0], cmap='gray', interpolation='none')
            plt.imshow(data.cpu().numpy()[i][0], interpolation='none')
            plt.title("Pre:{}".format(self.classes[predicted.cpu().numpy()[i]]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig(img_path + "_result.png")
        plt.show()



        accuracy = test_correct / len(self.test.dataset)
        loss = test_loss / len(self.test.dataset)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))
        print("test correctL:{0}, len of test:{1}".format( test_correct, len(self.test.dataset)))
        log_file.write("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy)+"\n")
        log_file.write("test correctL:{0}, len of test:{1}".format( test_correct, len(self.test.dataset))+"\n")

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))
        log_file.write("Time Usage: {:5.2f} mins.".format(time_diff / 60.)+"\n")
        log_file.close()
        self.print_all_classes_accuracy(classes_correct_dict, classes_all_num_dict)




    def identify_classes_accuracy(self, classes_correct_dict, classes_all_num_dict, predicted, label):
        """
        recognize the accuracy of each classes
        :param classes_correct_dict:
        :param predict:
        :param label:
        :return:
        """
        keys_list = list(classes_correct_dict.keys())
        predicted_list = predicted.view(-1)
        label_list = label.view(-1)
        for i in range(len(predicted_list)):
            if predicted_list[i] == label_list[i]:
                key_i = keys_list[predicted_list[i]]
                classes_correct_dict[key_i] += 1
            classes_all_num_dict[keys_list[label_list[i]]] += 1
        return classes_correct_dict, classes_all_num_dict

    def print_all_classes_accuracy(self, classes_corrext_dict, classes_all_num_dict):
        """
        print accuracy of each classes
        :param classes_corrext_dict:
        :param classes_all_num_dict:
        :return:
        """
        log_file = open(LOG_PATH + self.dataset_name + "_" + self.device_config + "_" + self.net_name + ".txt", "a")
        for key, value in classes_corrext_dict.items():
            print("The accuracy of ", str(key), ": ", value/classes_all_num_dict[key])
            log_file.write("The accuracy of "+str(key)+": "+str(value/classes_all_num_dict[key])+"\n")
        log_file.close()


if __name__ == "__main__":
    mp.freeze_support()

    for i in range(2,5):
        TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=i),
                 net=Net.ModifiedLeNet(input_channel=3, H=64, W=64),
                 # net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=Config.num_classes),
                 device_config=i)
        # TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=i),
        #          net=Net.ModifiedLeNet(input_channel=3, H=64, W=64),
        #          # net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=Config.num_classes),
        #          device_config=i)
        # TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=i),
        #          net=Net.mobilenet_v3_large(input_channel=3),
        #          # net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=Config.num_classes),
        #          device_config=i)
    # for i in range(4, 8):
    #     print(i)
    #     TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=1),
    #                  net=Net.ModifiedLeNet(input_channel=3, H=64, W=64),
    #                  # net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=Config.num_classes),
    #                  device_config=1, cpu_parallel_process=i)
    #     TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=1),
    #                  net=Net.mobilenet_v3_large(input_channel=3),
    #                  # net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=Config.num_classes),
    #                  device_config=1, cpu_parallel_process=i)
    #
    # TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=0),
    #              net=Net.LeNet(input_channel=3, H=32, W=32),
    #              # net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=Config.num_classes),
    #              device_config=0)

    # TrainProcess(dataLoader=DataLoader.CIFARLoader(device_config=1),
    #              net=Net.LeNet(input_channel=3, H=64, W=64),
    #              # net=Net.swin_tiny_patch4_window7_224(in_channel=3, num_classes=Config.num_classes),
    #              device_config=1, cpu_parallel_process=8)

