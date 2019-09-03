import os
import shutil
import numpy as np
import globalVar as gl
import re
import csv

n_steps = 16  # 时间步大小
train_rate = 0.6

# 读取txt数据并以列表形式保存
def read_file(file_path):
    data_lists = []
    with open(file_path, 'r') as file:
        datas = file.readlines()
        for line in datas:
            data_l = line.strip('\n').split(',')
            data_list = list(map(float, data_l))
            data_lists.append(data_list)
    return data_lists


# 读取切分后的数据
def read_split(file_path):
    split_lists = []
    with open(file_path) as file:
        split = file.readlines()
        for line in split:
            split_l = line.strip('\n')
            if split_l:
                split_list = int(split_l)
                split_lists.append(split_list)
    return split_lists


# 生成标签数据
def generat_data_label(lists_length, split_lists):
    empty_list = np.zeros((lists_length, 1), dtype=int)
    # try:
    for i in range(1, split_lists[0] + 1):
        start_i = i * 2 - 1
        end_i = i * 2
        empty_list[split_lists[start_i]:split_lists[end_i], :] = 1
    label_gen = empty_list.ravel()  # 将二维列表转换为一维
    return label_gen


def save_label(file_path, label_gen):
    with open(file_path, 'w') as f:
        for l in label_gen:
            f.write(str(l))
            f.write('\n')
        f.close()
    return f


# 批量读取文件，并把文件名放入列表
def batch_read_flies(data_path, path_filename, only_filename):
    for root, dirs, files in os.walk(data_path):
        for f in files:
            tmp = os.path.join(root, f)
            path_filename.append(tmp)
            only_filename.append(f)
    try:
        pass
    except (IndexError, IOError) as e:
        print("invalid file detected...")
        exit(1)
    return path_filename, only_filename


# 匹配两个相同文件名称的文件
def get_same_name(src_filename, split_filename_list):
    print('src_filename',src_filename)
    for split_filename in split_filename_list:
        p = re.compile(src_filename)
        match_result = re.findall(p, split_filename)
        if match_result:
            return match_result
    return ''


if __name__ == '__main__':
    gl._init()  # 文件初始化
    workspace_dir = gl.get_value("workdir")
    swim_split_dir = gl.get_value("swim_split")
    swim_raw_dir = gl.get_value("swim_raw")

    if (os.path.exists('out')):  # 若out文件夹已经存在，则进行删除
        print("remove old out")
        shutil.rmtree(workspace_dir + "out")  # 递归的方式删除里面的目录

    print("mkdir necessary dirs \n")  # 打印创建文件的路径，并在设定的相应的文件夹中创建out文件夹
    os.mkdir(workspace_dir + "out")
    os.mkdir(workspace_dir + "out/labelData")  # 存放带标签的文件
    # 把带标签的数据整个成一个文件，分为name,data,label三列用于RNN训练的数据
    os.mkdir(workspace_dir + "out/dataRNN")
    os.mkdir(workspace_dir + "out/trainRNNData")  # 用于训练的数据
    os.mkdir(workspace_dir + "out/testRNNData")  # 用于测试的数据

    # 遍历路径，找到原始数据文件swim_raw和swim_split文件
    swim_split_filename = []
    raw_swim_split_filename = []
    root_raw = swim_raw_dir  # 将原始数据路径放在一个列表中
    root_split = swim_split_dir  # 将切片数据路径放在一个列表中

    # 遍历原始数据文件夹，将文件路径和文件名分别存入raw_swim_raw_filename, swim_raw_filename
    path_swim_raw_filename, swim_raw_filename = batch_read_flies(
        root_raw, path_filename=[], only_filename=[])
    # 遍历切分数据文件夹，将文件路径和文件名分别存入raw_swim_raw_filename, swim_raw_filename
    path_swim_split_filename, swim_split_filename = batch_read_flies(
        root_split, path_filename=[], only_filename=[])
    trainRNNData = open('out/trainRNNData/trainRNNData.csv', 'a', newline='')
    testRNNData = open('out/testRNNData/testRNNData.csv', 'a', newline='')
    train_csv_write = csv.writer(trainRNNData)
    test_csv_write = csv.writer(testRNNData)
    count = 1
    for raw_filename_dir, raw_filename, split_filename_dir, split_filename in zip(path_swim_raw_filename, swim_raw_filename, path_swim_split_filename, swim_split_filename):
        swim_split_filename_c = get_same_name(
            raw_filename[:-4], swim_split_filename)
        batch_raw_datas = read_file(
            file_path=swim_raw_dir + "".join(swim_split_filename_c) + '.txt')
        batch_split_datas = read_split(
            file_path=swim_split_dir + "".join(swim_split_filename_c) + '.txt')
        gen_label = generat_data_label(len(batch_raw_datas), batch_split_datas)
        # TODO:在降采样前需要先过带通
        # 对batch_raw_datas和gen_label降采样，由50Hz变为5Hz：
        gen_label = gen_label[::10]
        batch_raw_datas = batch_raw_datas[::10]
        # 打开下面行则可以把label数据写入./out/labelData下
        # writer_label = save_label(file_path=workspace_dir + "out/labelData/" + "".join(swim_split_filename_c) + '.txt', label_gen=gen_label)
        for line in range(n_steps, len(batch_raw_datas)):
            row = []
            for col in range(3*(line - n_steps),3*line):
                row.append(batch_raw_datas[col//3][col%3])
            if gen_label[line-1] == 0:
                row.append(1)
                row.append(0)
            else:
                row.append(0)
                row.append(1)
            if count < len(batch_raw_datas)*train_rate:# 训练集与测试集初步区分
                train_csv_write.writerow(row)
            else:
                test_csv_write.writerow(row)
        print('当前处理文件:', count)
        count += 1
    trainRNNData.close()
    testRNNData.close()
