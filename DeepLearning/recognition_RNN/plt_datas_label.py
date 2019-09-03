import os
import shutil
import numpy as np
import globalVar as gl
import re
import matplotlib.pyplot as plt

#读取txt数据并以列表形式保存
def read_file(file_path):
    data_lists = []
    with open(file_path, 'r') as file:
        datas = file.readlines()
        for line in datas:
            data_l = line.strip('\n').split(',')
            data_list = list(map(float, data_l))
            data_lists.append(data_list)
            #print(data_list)
    return data_lists

#读取切分后的数据
#def read_split(file_path, file_name):
def read_split(file_path):
    split_lists = []
    with open(file_path) as file:
        split = file.readlines()
        for line in split:
            split_l = line.strip('\n')
            if split_l:
                split_list = int(split_l)
                split_lists.append(split_list)
        #split_list = map(int, split_lists)
    split_lists1 = np.ravel(split_lists)     #将二维列表转换为一维
    return split_lists1
    #split_lists = _flatten(split_lists)
    #fist = split_lists[0]
    #print(fist)

    return  split_lists

#生成标签数据
def generat_data_label(data_lists, split_lists):
    lists_length = len(data_lists)
    empty_list = np.zeros((lists_length, 1), dtype=int)
    #try:
    for i in range(1, split_lists[0] + 1):
        start_i = i * 2 - 1
        end_i = i * 2
        empty_list[split_lists[start_i]:split_lists[end_i],:] = 1
        label_gen = empty_list.ravel()  #将二维列表转换为一维
    #except :
     #   pass
    #label_gen = _flatten(empty_list)
    #print(label_gen)
    return label_gen

def save_label(file_path, label_gen):
    #label_gen.ravel()
    with open(file_path, 'w') as f:

        for l in label_gen:
            f.write(str(l))
            f.write('\r\n')
        f.close()
    #f = np.savetxt(file_path + file_name, label_gen, delimiter=',')
    return f

#批量读取文件，并把文件名放入列表
def batch_read_flies(data_path, path_filename, only_filename):
    for root, dirs, files in os.walk(data_path):
        for f in files:
            tmp = os.path.join(root, f)
            if ("泳" in tmp):
                path_filename.append(tmp)
                only_filename.append(f)
    try:
        # user enters in the filename of the csv file to convert
        # in_filename = argv[1:]
        print("data_rnn.py received list :" + str(path_filename))
    except (IndexError, IOError) as e:
        print("invalid file detected...")
        exit(1)
    # print(path_filename)
    # print(only_filename)
    return path_filename, only_filename

#匹配两个相同文件名称的文件
def get_same_name(src_filename, split_filename_list):
    for split_filename in split_filename_list:
        p = re.compile(src_filename)
        # match_result  = pattern.findall('abcd')
        match_result = re.findall(p, split_filename)
        if match_result:
            return match_result

#sigmoid函数将数据归一化
def sigmoid_1(x):
    s = 1/(1+np.exp(x))
    return s

if __name__ == '__main__':
    gl._init()  # 文件初始化
    workspace_dir = gl.get_value("workdir")  # DataRNN存放位置C:\\Users\\Administrator\\PycharmProjects\\swim_lstm\\workspace\\
    swim_raw_dir = gl.get_value("swim_raw")
    swim_label = gl.get_value("swim_label")
    file_name = "trans_游泳-仰泳-女-向紫韵20180720_15_31_45.txt"


    #原始数据
    raw_datas = read_file(file_path=swim_raw_dir+file_name)
    raw_datas_x = np.array(raw_datas)[:, 0]
    raw_datas_y = np.array(raw_datas)[:, 1]
    raw_datas_z = np.array(raw_datas)[:, 2]
    #raw_datas = np.ravel(raw_datas)
    #标签数据
    label_datas = read_split(file_path=swim_label+file_name)

    x_axis = np.arange(0, len(label_datas), 1)

    plt.figure()
    plt.plot( x_axis, raw_datas_x)
    #plt.plot(x_axis, raw_datas_y, color='green')
    #plt.plot(x_axis, raw_datas_z, color='yellow')
    plt.plot(x_axis, label_datas, color='red')
    plt.show()
    print(raw_datas_x)
    print(label_datas)



    # if (os.path.exists('out')):  # 若out文件夹已经存在，则进行删除
    #     print("remove old out")
    #     shutil.rmtree(workspace_dir + "out")  # 递归的方式删除里面的目录
    #
    # print("mkdir necessary dirs \n")  # 打印创建文件的路径，并在设定的相应的文件夹中创建out文件夹
    # os.mkdir(workspace_dir + "out")
    # os.mkdir(workspace_dir + "out\\labelData")  # 存放带标签的文件
    # os.mkdir(workspace_dir + "out\\dataRNN")  # 把带标签的数据整个成一个文件，分为name,data,label三列用于RNN训练的数据
    # os.mkdir(workspace_dir + "out\\trainRNNData")  # 用于训练的数据
    # os.mkdir(workspace_dir + "out\\testRNNData")  # 用于测试的数据
    #
    # # 遍历路径，找到原始数据文件swim_raw和swim_split文件
    # # swim_raw_filename = []
    # # raw_swim_raw_filename = []
    # swim_split_filename = []
    # raw_swim_split_filename = []
    # root_raw = swim_raw_dir  # 将原始数据路径放在一个列表中
    # root_split = swim_split_dir  # 将切片数据路径放在一个列表中
    #




    # #遍历原始数据文件夹，将文件路径和文件名分别存入raw_swim_raw_filename, swim_raw_filename
    # path_swim_raw_filename, swim_raw_filename = batch_read_flies(root_raw, path_filename=[], only_filename=[])
    # # print(path_swim_raw_filename)
    # # print(swim_raw_filename)
    # # 遍历切分数据文件夹，将文件路径和文件名分别存入raw_swim_raw_filename, swim_raw_filename
    # path_swim_split_filename, swim_split_filename = batch_read_flies(root_split, path_filename=[], only_filename=[])
    # # print(path_swim_split_filename)
    # #print(swim_split_filename)
    #
    #
    # for raw_filename_dir, raw_filename, split_filename_dir, split_filename in zip(path_swim_raw_filename, swim_raw_filename, path_swim_split_filename, swim_split_filename):
    #     swim_split_filename_c = get_same_name(raw_filename[:-4], swim_split_filename)
    #     #print(swim_split_filename_c)
    #     batch_raw_datas = read_file(file_path=swim_raw_dir + "".join(swim_split_filename_c) + '.txt')
    #     batch_split_datas = read_split(file_path=swim_split_dir + "".join(swim_split_filename_c) + '_split_index.txt')
    #     gen_label = generat_data_label(batch_raw_datas, batch_split_datas)
    #     writer_label = save_label(file_path=workspace_dir + "out\\labelData\\" + "".join(swim_split_filename_c) + '.txt', label_gen=gen_label)
    #     #print(batch_split_datas)