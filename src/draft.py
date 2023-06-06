#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   draft.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/3/17 23:06
------------      --------    -----------

"""
from datetime import datetime

from matplotlib import pyplot as plt


def train_on_gpu_and_cpu(self):
    log_file = open(LOG_PATH + self.dataset_name + "_" + self.device_config + "_" + self.net_name + ".txt", "w")
    img_path = IMAGE_DIR + self.dataset_name + "_" + self.device_config + "_" + self.net_name
    steps = 0
    start_time = datetime.now()
    loss_list_train = []
    loss_list_line = []
    accuracy_list_train = []
    accuracy_list_line = []
    artists = []
    train_start_time = datetime.now()
    fig, ax = plt.subplots()
    line_loss = ax.plot([i for i in range(len(loss_list_train))], loss_list_train, label='loss',
                        c='blue', ls='-')
    line_acc = ax.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train, label='accuracy', c='r',
                       ls='-')

    print("Training & Evaluating......")
    for epoch in range(Config.epoch):

        print("Epoch {:3}".format(epoch + 1))
        for batch_idx, (data, label) in tqdm(enumerate(self.train), total=len(self.train),
                                             desc='Epoch {}'.format(epoch)):
            # for data, label in self.train:
            # cpu --> gpu
            data, label = data.to(device), label.to(device)
            # cpu-->gpu
            self.optimizer.zero_grad()
            outputs = self.net(data)
            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()  # update the arguments by grad

            # print results ever 100 times
            if steps % Config.print_per_step == 0:
                _, predicted = torch.max(outputs, 1)
                correct = int(sum(predicted == label))
                accuracy = correct / Config.batch_size  # calculate the accuracy
                end_time = datetime.now()
                time_diff = (end_time - start_time).seconds
                time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                log_file.write(msg.format(steps, loss, accuracy, time_usage) + "\n")
                if steps % Config.plt_per_step == 0:
                    loss_list_train.append(loss.item())
                    accuracy_list_train.append(accuracy)
                    line_loss = ax.plot([i for i in range(len(loss_list_train))], loss_list_train,
                                        c='blue', ls='-')
                    line_acc = ax.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train,
                                       c='r',
                                       ls='-')
                    show_time = [
                        ax.text(Config.plt_time_loc_x, Config.plt_time_loc_y, s="train time:" + str(time_usage),
                                ha='right')]
                    accuracy_list_line.append(line_acc)

                    loss_list_line.append(line_loss)
                    artists.append([line_acc, line_loss, show_time])
                    plt.pause(0.1)
            self.scheduler.step()  # update learning rate

            steps += 1

    train_end_time = datetime.now()
    train_time_diff = (train_end_time - train_start_time).seconds
    time_usage = '{:3}m{:3}s'.format(int(train_time_diff / 60), time_diff % 60)
    msg = " Time usage of train:{:9}."
    log_file.write(msg.format(time_usage) + "\n")

    ax.legend(loc='upper right')

    """
    Draw the diagram of accuracy and loss in train
    """
    train_times = np.linspace(1, len(loss_list_train), len(loss_list_train))
    plt.plot(train_times, loss_list_train)
    plt.title("Loss of CPU training processs")
    plt.show()
    plt.savefig(img_path + "_loss.png")
    plt.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train)
    plt.title("Accuracy of CPU training process ")
    plt.show()
    plt.savefig(img_path + "_accuracy.png")
    log_file.close()

    self.draw_train_mp4(fig, artists)



    def train_on_cpu(self):
        log_file = open(LOG_PATH + self.dataset_name + "_" + self.device_config + "_" + self.net_name + ".txt", "w")
        img_path = IMAGE_DIR + self.dataset_name + "_" + self.device_config + "_" + self.net_name
        steps = 0
        start_time = datetime.now()
        loss_list_train = []
        loss_list_line = []
        accuracy_list_train = []
        accuracy_list_line = []
        artists = []
        train_start_time = datetime.now()
        fig, ax = plt.subplots()
        line_loss = ax.plot([i for i in range(len(loss_list_train))], loss_list_train, label='loss',
                            c='blue', ls='-')
        line_acc = ax.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train, label='accuracy', c='r',
                           ls='-')

        print("Training & Evaluating......")
        for epoch in range(Config.epoch):

            print("Epoch {:3}".format(epoch + 1))
            for batch_idx, (data, label) in tqdm(enumerate(self.train), total=len(self.train),
                                                 desc='Epoch {}'.format(epoch)):
                self.optimizer.zero_grad()
                outputs = self.net(data)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()  # update the arguments by grad

                # print results ever 100 times
                if steps % Config.print_per_step == 0:
                    _, predicted = torch.max(outputs, 1)
                    correct = int(sum(predicted == label))
                    accuracy = correct / Config.batch_size  # calculate the accuracy
                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    log_file.write(msg.format(steps, loss, accuracy, time_usage) + "\n")
                    if steps % Config.plt_per_step == 0:
                        loss_list_train.append(loss.item())
                        accuracy_list_train.append(accuracy)
                        line_loss = ax.plot([i for i in range(len(loss_list_train))], loss_list_train,
                                            c='blue', ls='-')
                        line_acc = ax.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train,
                                           c='r',
                                           ls='-')
                        show_time = [
                            ax.text(Config.plt_time_loc_x, Config.plt_time_loc_y, s="train time:" + str(time_usage),
                                    ha='right')]
                        accuracy_list_line.append(line_acc)

                        loss_list_line.append(line_loss)
                        artists.append([line_acc, line_loss, show_time])
                        plt.pause(0.1)
                self.scheduler.step()  # update learning rate

                steps += 1

        train_end_time = datetime.now()
        train_time_diff = (train_end_time - train_start_time).seconds
        time_usage = '{:3}m{:3}s'.format(int(train_time_diff / 60), time_diff % 60)
        msg = " Time usage of train:{:9}."
        log_file.write(msg.format(time_usage) + "\n")

        ax.legend(loc='upper right')

        """
        Draw the diagram of accuracy and loss in train
        """
        train_times = np.linspace(1, len(loss_list_train), len(loss_list_train))
        plt.plot(train_times, loss_list_train)
        plt.title("Loss of CPU training processs")
        plt.show()
        plt.savefig(img_path + "_loss.png")
        plt.plot([i for i in range(len(accuracy_list_train))], accuracy_list_train)
        plt.title("Accuracy of CPU training process ")
        plt.show()
        plt.savefig(img_path + "_accuracy.png")
        log_file.close()

        self.draw_train_mp4(fig, artists)



