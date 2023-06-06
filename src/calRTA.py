#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   calRTA.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/4/1 17:34
------------      --------    -----------

"""
import math
import os
import re

acc_list = []
time_list = []
RTA_list = []
Lenet_pattern = re.compile("^.*Lenet.*$")
ModifiedLenet_pattern = re.compile("^.*ModifiedLeNet.*$")
MobileNetV3_pattern = re.compile("^.*MobileNetV3.*$")
ViT_pattern = re.compile("^.*ViT.*$")
log_dir = "../out_A4000/log/"

for filename in os.listdir(log_dir):
        if filename.endswith(".txt") and filename.startswith("CIFAR"):
            # and (not gpu_and_cpu_pattern.match(filename)):

            # gpu_and_cpu_pattern.match(filename) or gpu_parallel_pattern.match(filename) or only_gpu_pattern.match(filename)):
            file_path = os.path.join(log_dir, filename)
            file = open(file_path, "r")
            filename = filename.split(".")[0].split("CIFAR_")[1]
            # filename = filename.split("_ViT")[0]
            print(filename)
            if ViT_pattern.match(filename):
                filename = filename.split("_")[0]
                while True:
                    line = file.readline()
                    if " Time usage of train:" in line:
                        line = line.replace(" ", "")
                        print(line)
#                        acc = float(line.split("Accuracy:")[1].split("%")[0])
                        time = line.split("train:")[1]
                        minute = int(time.split("m")[0])
                        second = int(time.split('m')[1].split("s")[0])

                        time_list.append(minute*60+second)

                    if "Test Loss:" in line:
                        line = line.replace(" ", "")
                        print(line)
                        acc = float(line.split("Accuracy:")[1].split("%")[0])
                        acc_list.append(acc)
                        print("the accuracy of ", filename, " is ", acc)
                    if not line:
                        break

for i in range(len(acc_list)):
    RTA_list.append(-math.log(time_list[i]/time_list[0] * (acc_list[0]/acc_list[i])))


print(time_list)
print(acc_list)
print(RTA_list)