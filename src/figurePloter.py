#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   figurePloter.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/3/26 21:57
------------      --------    -----------

"""
import os
import re

import matplotlib.pyplot as plt

# log_dir = "../out/cpu_parallel/gpu_parallel_log/"
log_dir = "../out_GTX1070Ti/log/"

# log_dir = "../out_GTX1060/log/"
only_cpu_pattern = re.compile("^(?!.*gpu).*_cpu_(?!.*parallel).*\.txt$")
cpu_parallel_pattern = re.compile("^.*_cpu_parallel.*\.txt$")
gpu_and_cpu_pattern = re.compile("^.*_gpu_and_cpu_.*\.txt$")
only_gpu_pattern = re.compile("^(?!.*cpu).*_gpu_(?!.*parallel).*\.txt$")
gpu_parallel_pattern = re.compile("^.*_gpu_data_parallel.*\.txt$")
Lenet_pattern = re.compile("^.*Lenet.*$")
ModifiedLenet_pattern = re.compile("^.*ModifiedLeNet.*$")
MobileNetV3_pattern = re.compile("^.*MobileNetV3.*$")
ViT_pattern = re.compile("^.*ViT.*$")
# for file in os.listdir(log_dir):
#     if Lenet_pattern.match(file) and only_cpu_pattern.match(file):
#         print(str(file))

# time_dict = {}
# for filename in os.listdir(log_dir):
#     if filename.endswith(".txt") and Lenet_pattern.match(filename) and only_cpu_pattern.match(filename):
#         file_path = os.path.join(log_dir, filename)
#         file = open(file_path, "r")
#         filename = filename.split(".")[0].split("Lenet_cpu_")[1]
#         while True:
#             line = file.readline()
#             if "Time usage of train:" in line:
#                 line = line.replace(" ","")
#                 print(line)
#                 time = line.split("train:")[1]
#                 minute = int(time.split("m")[0])
#                 second = int(time.split('m')[1].split("s")[0])
#                 time_dict[filename] = minute*60+second
#                 print("the time of ",filename," is ",minute*60+second,"s")
#             if not line:
#                 break

time_dict_parallel = {}
time_dict_gpu = {}
acc_ViT = {}
time_ViT = {}
out_log = open(str(log_dir)+".txt","w")
for filename in os.listdir("../out_A4000/log/"):
        if filename.endswith(".txt") and filename.startswith("CIFAR"):
            # and (not gpu_and_cpu_pattern.match(filename)):

            # gpu_and_cpu_pattern.match(filename) or gpu_parallel_pattern.match(filename) or only_gpu_pattern.match(filename)):
            file_path = os.path.join("../out_A4000/log/", filename)
            file = open(file_path, "r")
            filename = filename.split(".")[0].split("CIFAR_")[1]
            # filename = filename.split("_ViT")[0]
            print(filename)
            if ViT_pattern.match(filename):
                filename = filename.split("_ViT")[0]
                while True:
                    line = file.readline()
                    if " Time usage of train:" in line:
                        line = line.replace(" ", "")
                        print(line)
#                        acc = float(line.split("Accuracy:")[1].split("%")[0])
                        time = line.split("train:")[1]
                        minute = int(time.split("m")[0])
                        second = int(time.split('m')[1].split("s")[0])

                        time_ViT[filename] = minute*60+second

                    if "Test Loss:" in line:
                        line = line.replace(" ", "")
                        print(line)
                        acc = float(line.split("Accuracy:")[1].split("%")[0])
                        # time = line.split("train:")[1]
                        # minute = int(time.split("m")[0])
                        # second = int(time.split('m')[1].split("s")[0])

                        # acc_lenet[filename] = minute*60+second
                        acc_ViT[filename] = acc
                        print("the accuracy of ", filename, " is ", acc)
                    if not line:
                        break
plt.figure(figsize=(10,8))
plt.bar(time_ViT.keys(), time_ViT.values(), color="#4c72b0", width=0.25, label="ViT")
plt.title("Time of accelerated Strategies on ViT of CIFAR")
plt.legend(loc="upper right")
plt.savefig("../out_A4000/log/"+"_time_ViT_CIFAR.png")
plt.show()

plt.figure(figsize=(10,8))
plt.plot(acc_ViT.keys(), acc_ViT.values(), color="#c44e52", marker="o", label="ViT")
plt.title("Accuracy of accelerated Strategies on ViT of CIFAR")
plt.legend(loc="upper right")
plt.savefig("../out_A4000/log/" + "_acc_ViT_CIFAR.png")
plt.show()



def draw_acc():
    acc_lenet = {}
    acc_ModifiedLenet = {}
    acc_MobileNet = {}
    time_ViT = {}
    for filename in os.listdir(log_dir):
        if filename.endswith(".txt") and filename.startswith("CIFAR"):
            # and (not gpu_and_cpu_pattern.match(filename)):

            # gpu_and_cpu_pattern.match(filename) or gpu_parallel_pattern.match(filename) or only_gpu_pattern.match(filename)):
            file_path = os.path.join(log_dir, filename)
            file = open(file_path, "r")
            filename = filename.split(".")[0].split("CIFAR_")[1]
            # filename = filename.split("_ViT")[0]
            print(filename)
            # while True:
            #     line = file.readline()
            #     if " Time usage of train:" in line:
            #         line = line.replace(" ","")
            #         print(line)
            #         # acc = float(line.split("Accuracy:")[1].split("%")[0])
            #         time = line.split("train:")[1]
            #         minute = int(time.split("m")[0])
            #         second = int(time.split('m')[1].split("s")[0])
            #         # time_dict_gpu[filename] = minute*60+second
            #         time_ViT[filename] = minute*60+second
            #         # print("the accuracy of ",filename," is ",acc)
            #     if not line:
            #         break

            if Lenet_pattern.match(filename):
                filename = filename.split("_Lenet")[0]
                while True:
                    line = file.readline()
                    # if " Time usage of train:" in line:
                    if "Test Loss:" in line:
                        line = line.replace(" ", "")
                        print(line)
                        acc = float(line.split("Accuracy:")[1].split("%")[0])
                        # time = line.split("train:")[1]
                        # minute = int(time.split("m")[0])
                        # second = int(time.split('m')[1].split("s")[0])

                        # acc_lenet[filename] = minute*60+second
                        acc_lenet[filename] = acc
                        print("the accuracy of ", filename, " is ", acc)
                    if not line:
                        break
            elif ModifiedLenet_pattern.match(filename):
                filename = filename.split("_ModifiedLeNet")[0]
                while True:
                    line = file.readline()
                    # if " Time usage of train:" in line:
                    if "Test Loss:" in line:
                        line = line.replace(" ", "")
                        print(line)
                        acc = float(line.split("Accuracy:")[1].split("%")[0])
                        # time = line.split("train:")[1]
                        # minute = int(time.split("m")[0])
                        # second = int(time.split('m')[1].split("s")[0])

                        # acc_ModifiedLenet[filename] = minute*60+second
                        acc_ModifiedLenet[filename] = acc
                        print("the accuracy of ", filename, " is ", acc)
                    if not line:
                        break
            elif MobileNetV3_pattern.match(filename):
                filename = filename.split("_MobileNetV3")[0]
                print(filename)
                while True:
                    line = file.readline()
                    if "Test Loss:" in line:
                        line = line.replace(" ", "")
                        print(line)
                        acc = float(line.split("Accuracy:")[1].split("%")[0])
                        # time = line.split("train:")[1]
                        # minute = int(time.split("m")[0])
                        # second = int(time.split('m')[1].split("s")[0])

                        # acc_MobileNet[filename] = minute*60+second
                        acc_MobileNet[filename] = acc
                        print("the accuracy of ", filename, " is ", acc)
                    if not line:
                        break

    # colors = ["#4c72b0" if key != "gpu" else "#c44e52" for key in time_dict.keys()]
    bar_width = 0.22
    plt.figure(figsize=(10, 8))

    # plt.ylim(700, 1800)
    labels = list(time_dict_gpu.keys())

    x = range(len(labels))
    out_log.write("acc lenet:"+str(acc_lenet)+"\nacc ModifiedLenet:"+str(acc_ModifiedLenet)+"\nacc MobileNetV3:"+str(acc_MobileNet)+"\n")
    # plt.plot(time_ViT.keys(), time_ViT.values(), color="#4c72b0", marker="o",label="MobileNetV3")
    plt.plot(acc_lenet.keys(), acc_lenet.values(), color="#4c72b0", marker="o", label="LeNet-5")
    plt.plot(acc_ModifiedLenet.keys(), acc_ModifiedLenet.values(), color="#c44e52", marker="o", label="Modified LeNet")
    plt.plot(acc_MobileNet.keys(), acc_MobileNet.values(), color="#55a868", marker="o", label="MobileNetV3")
    # plt.plot([i + bar_width for i in x], time_dict_parallel.values(), color="#c44e52", width=0.2, label="gpu data parallel")

    # for bar in gpu_parallel_bar + gpu_bar:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #              '%d' % int(height),
    #              ha='center', va='bottom')
    # plt.yticks([i for i in range(70,100,5)])
    plt.legend(loc="upper right")
    plt.title("Accuracy of GPU and GPU data parallel in CIFAR")
    plt.ylabel("%")
    # plt.xlabel("Net")
    # plt.xticks([i + bar_width/2 for i in x], labels)
    # plt.xlabel("process of cpu parallelu")
    plt.savefig(log_dir + "acc_gpu_parallel_time_CIFAR.png")
    plt.show()



def draw_time():
    time_lenet = {}
    time_ModifiedLenet = {}
    time_MobileNet = {}
    time_ViT = {}
    for filename in os.listdir(log_dir):
        if filename.endswith(".txt") and filename.startswith("CIFAR"):
            # and (not gpu_and_cpu_pattern.match(filename)):

            # gpu_and_cpu_pattern.match(filename) or gpu_parallel_pattern.match(filename) or only_gpu_pattern.match(filename)):
            file_path = os.path.join(log_dir, filename)
            file = open(file_path, "r")
            filename = filename.split(".")[0].split("CIFAR_")[1]
            # filename = filename.split("_ViT")[0]
            print(filename)
            # while True:
            #     line = file.readline()
            #     if " Time usage of train:" in line:
            #         line = line.replace(" ","")
            #         print(line)
            #         # acc = float(line.split("Accuracy:")[1].split("%")[0])
            #         time = line.split("train:")[1]
            #         minute = int(time.split("m")[0])
            #         second = int(time.split('m')[1].split("s")[0])
            #         # time_dict_gpu[filename] = minute*60+second
            #         time_ViT[filename] = minute*60+second
            #         # print("the accuracy of ",filename," is ",acc)
            #     if not line:
            #         break

            if Lenet_pattern.match(filename):
                filename = filename.split("_Lenet")[0]
                while True:
                    line = file.readline()
                    if " Time usage of train:" in line:

                        line = line.replace(" ", "")
                        print(line)

                        time = line.split("train:")[1]
                        minute = int(time.split("m")[0])
                        second = int(time.split('m')[1].split("s")[0])

                        time_lenet[filename] = minute*60+second

                    if not line:
                        break
            elif ModifiedLenet_pattern.match(filename):
                filename = filename.split("_ModifiedLeNet")[0]
                while True:
                    line = file.readline()
                    if " Time usage of train:" in line:

                        line = line.replace(" ", "")


                        time = line.split("train:")[1]
                        minute = int(time.split("m")[0])
                        second = int(time.split('m')[1].split("s")[0])

                        time_ModifiedLenet[filename] = minute*60+second

                    if not line:
                        break
            elif MobileNetV3_pattern.match(filename):
                filename = filename.split("_MobileNetV3")[0]
                print(filename)
                while True:
                    line = file.readline()
                    if " Time usage of train:" in line:
                        line = line.replace(" ", "")
                        print(line)

                        time = line.split("train:")[1]
                        minute = int(time.split("m")[0])
                        second = int(time.split('m')[1].split("s")[0])

                        time_MobileNet[filename] = minute*60+second

                    if not line:
                        break

    # colors = ["#4c72b0" if key != "gpu" else "#c44e52" for key in time_dict.keys()]
    bar_width = 0.35

    # plt.ylim(700, 1800)
    labels = list(time_lenet.keys())
    plt.figure(figsize=(10, 8))

    x = range(len(labels))
    out_log.write(
        "time lenet:" + str(time_lenet) + "\ntime ModifiedLenet:" + str(time_ModifiedLenet) + "\ntime MobileNetV3:" + str(
            time_MobileNet) + "\n")
    # plt.plot(time_ViT.keys(), time_ViT.values(), color="#4c72b0", marker="o",label="MobileNetV3")
    plt.bar([i for i in x], time_lenet.values(), color="#4c72b0", width=0.25, label="LeNet-5")
    plt.bar([i + bar_width for i in x], time_ModifiedLenet.values(), color="#c44e52",  width=0.25, label="Modified LeNet")
    plt.bar([i + 2 * bar_width for i in x], time_MobileNet.values(), color="#55a868", width=0.25,label="MobileNetV3")
    # plt.bar([i + bar_width for i in x], time_dict_parallel.values(), color="#c44e52", width=0.2, label="gpu data parallel")

    # for bar in gpu_parallel_bar + gpu_bar:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #              '%d' % int(height),
    #              ha='center', va='bottom')
    # plt.yticks([i for i in range(70,100,5)])
    plt.legend(loc="upper right")
    plt.title("Time of GPU and GPU data parallel in CIFAR")
    plt.ylabel("%")
    # plt.xlabel("Net")
    plt.xticks([i + bar_width for i in x], labels)
    # plt.xlabel("process of cpu parallelu")
    plt.savefig(log_dir + "acc_gpu_parallel_time_CIFAR.png")
    plt.show()

draw_acc()
draw_time()
#
#
# plt.ylim(70, 90)
# pic_lenet = {"cpu":73.21, "cpu_parallel":73.46,"gpu":77.2,"gpu_data_parallel":77.77}
# pic_MobileNet = {"cpu":82.72, "cpu_parallel":84.46,"gpu":82.36,"gpu_data_parallel":82.32} #gpu and cpu:82.90
# pic_ModifiedLeNet = {"cpu":79.55, "cpu_parallel":80.79,"gpu": 79.38,"gpu_data_parallel":79.20} # cpu and gpu : 79.96
# plt.plot(pic_lenet.keys(), pic_lenet.values(), color="#4c72b0",  marker="o",label="LeNet-5")
# plt.plot(pic_ModifiedLeNet.keys(), pic_ModifiedLeNet.values(), color="#c44e52", marker="o",label="Modified LeNet")
# plt.plot(pic_MobileNet.keys(), pic_MobileNet.values(), color="#55a868", marker="o",label="MobileNetV3")
# plt.legend(loc="upper right")
# plt.title("Accuracy of GPU and GPU data parallel")
# plt.ylabel("accuracy(%)")
# plt.savefig(log_dir+"_c_cp_g_gdp_accuracy.png")
# plt.show()
#
# plt.ylim(70, 90)
# time_lenet = {"cpu":73.21, "cpu_parallel":73.46,"gpu":77.2,"gpu_data_parallel":77.77}
# time_MobileNet = {"cpu":82.72, "cpu_parallel":84.46,"gpu":82.36,"gpu_data_parallel":82.32} #gpu and cpu:82.90
# time_ModifiedLeNet = {"cpu":79.55, "cpu_parallel":80.79,"gpu": 79.38,"gpu_data_parallel":79.20} # cpu and gpu : 79.96
# plt.plot(pic_lenet.keys(), pic_lenet.values(), color="#4c72b0",  marker="o",label="LeNet-5")
# plt.plot(pic_ModifiedLeNet.keys(), pic_ModifiedLeNet.values(), color="#c44e52", marker="o",label="Modified LeNet")
# plt.plot(pic_MobileNet.keys(), pic_MobileNet.values(), color="#55a868", marker="o",label="MobileNetV3")
# plt.legend(loc="upper right")
# plt.title("Accuracy of GPU and GPU data parallel")
# plt.ylabel("accuracy(%)")
# plt.savefig(log_dir+"_c_cp_g_gdp_accuracy.png")
# plt.show()