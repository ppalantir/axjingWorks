import os
import re
import csv
from collections import Counter
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 


jpg_path ='/home/andy/dataVOC/JPEGImage'

def read_csv(csv_path):
    totals = []
    peoples = []
    with open(csv_path) as csv_files:
        dict_reader = csv.DictReader(csv_files)
        for row in dict_reader:
            totals.append(row['class'])
            #print(row['class'])
        
            file_name = re.split('-', row['filename'])
            people_name = file_name[0]
            peoples.append(people_name)
    dict_peoples = Counter(peoples)
    people_l = dict_peoples.keys()
    num_people = len(people_l)
    # print(dict_peoples)
    # print(people_l)
    # print(num_people)
    # print(totals, len(totals))
    return totals, num_people

def statistic_labels():
    # 统计斑块及硬化数量
    train_csv_path='/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/workspace/training_inception_v2_2/annotations/train_labels.csv'
    test_csv_path='/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/workspace/training_inception_v2_2/annotations/test_labels.csv'
    train_totals, train_people = read_csv(train_csv_path)
    test_totals, test_people = read_csv(test_csv_path)
    labels_totals = train_totals + test_totals
    num_people = train_people + test_people
    labels_num = len(labels_totals)
    
    plaque_list = []
    sclerosis_list = []
    normal_list = []
    pseudomorphism_list = []
    vessel_list = []
    for i in labels_totals:
        if i == 'plaque':
            plaque_list.append(i)
            plaque_num = len(plaque_list)
        elif i == 'sclerosis':
            sclerosis_list.append(i)
            sclerosis_num = len(sclerosis_list)
        elif i == 'normal':
            normal_list.append(i)
            normal_num = len(normal_list)
        elif i == 'pseudomorphism':
            pseudomorphism_list.append(i)
            pseudomorphism_num = len(pseudomorphism_list)
        # elif i == 'vessel':
        #     vessel_list.append(i)
        #     vessel_num = len(vessel_list)
    #提供汉字支持
    mpl.rcParams[u'font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    width=0.5
    x_ = ['num_people', 'num_label', 'plaque', 'sclerosis', 'normal', 'pseudomorphism']
    x_1 = np.arange(len(x_))
    
    y = [num_people, labels_num, plaque_num, sclerosis_num, normal_num, pseudomorphism_num]
    y_list = np.array(y)
    #y_single_list = np.array(y)/2
    plt.bar(x_1[0], y_list[0], width=width, color='red')
    plt.bar(x_1[1:], y_list[1:], width=width, color='green')
    #plt.bar(x_1+0.35, y_single_list, width =width,facecolor = 'yellowgreen',edgecolor = 'white')
    for x,y in zip(x_1,y_list):
        plt.text(x, y+width, y, fontsize=16, ha='center', va= 'bottom')
        #plt.text(x, y+width, '%.2f'%y, ha='center', va= 'bottom')
    # for x,y in zip(x_1,y_single_list):
    #     plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    #plt.rcParams['savefig.dpi'] = 800 #图片像素
    #plt.rcParams['figure.dpi'] = 800 #分辨率
    plt.xlabel('Sample classification')
    plt.ylabel('Number of samples')
    # plt.xticks((0, 1, 2, 3, 4, 5, 6), ('total_label', 'plaque', 'sclerosis', 'normal', 'pseudomorphism'))
    #plt.xticks((0.5,1.5,2.5,3.5), ('single_t', 'single_p', 'single_s', 'single_n'))
    plt.xticks((0, 1, 2, 3, 4, 5, 6), (x_[0], x_[1], x_[2], x_[3], x_[4], x_[5]))
    plt.title('Sample Statistic')
    plt.legend()
    plt.show()
          

if __name__ == '__main__':
    #statistic_analy()
    # train_csv_path='/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/workspace/training_inception_v2_1/annotations/train_labels.csv'
    # test_csv_path='/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/workspace/training_inception_v2_1/annotations/test_labels.csv'
    # train_totals = read_csv(train_csv_path)
    # test_totals = read_csv(test_csv_path)
    # #labels_totals = train_totals.append(test_totals)
    # labels_totals = train_totals + test_totals
    # print(labels_totals, len(labels_totals))
    statistic_labels()
