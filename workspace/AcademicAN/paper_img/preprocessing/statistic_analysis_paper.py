import os
import re
import csv
from collections import Counter
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 

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
    return totals, num_people

def statistic_labels():
    # 统计斑块及硬化数量
    train_csv_path='./paper_statistic.csv'
    #test_csv_path='./test_labels.csv'
    train_totals, train_people = read_csv(train_csv_path)
    #test_totals, test_people = read_csv(test_csv_path)
    labels_totals = train_totals
    num_people = train_people
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
        # elif i == 'pseudomorphism':
        #     pseudomorphism_list.append(i)
        #     pseudomorphism_num = len(pseudomorphism_list)
        elif i == 'vessel' or i == 'blood vessel':
            vessel_list.append(i)
            vessel_num = len(vessel_list)
            print(vessel_num)
    #提供汉字支持
    mpl.rcParams[u'font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    width=0.3
    x_ = ['人数', 'label总数', "vessel", 'plaque', 'sclerosis', 'normal']
    x_1 = np.arange(len(x_))
    
    y = [int(num_people), int((labels_num+1)/2), int(vessel_num/2), int((plaque_num+1)/2), int((sclerosis_num+1)/2), int(normal_num/2)]
    y_list = np.array(y)
    y_list2 = np.array(y)
    #y_single_list = np.array(y)/2
    plt.figure(figsize=(9, 6))
    plt.bar(x_1[0], y_list[0], width=width, color='red', label='人数')
    #plt.text(0, y_list[0]+width, y_list[0], fontsize=16, ha='center', va= 'bottom')
    plt.bar(x_1[1:], y_list[1:], width=width, alpha=0.8, color='green', label='横切')
    #plt.bar(x_1+0.35, y_single_list, width =width,facecolor = 'yellowgreen',edgecolor = 'white')
    for x,y in zip(x_1[:],y_list):
        print(x,y)
        if x==0:
            plt.text(x, y+width, y, fontsize=12, ha='center', va= 'bottom')
        else:
            plt.text(x, y+width, y, fontsize=12, ha='left', va= 'bottom')
    
    #plt.bar(x_1[0], y_list2[0], width=width, color='red')
    plt.bar(x_1[1:]+0.3, y_list2[1:], width=width, alpha=0.8, color='b', label='纵切')
    
    # plt.xlabel('Sample classification', fontsize=12)
    plt.ylabel('Number of samples', fontsize=12)
    
    plt.xticks((0, 1, 2, 3, 4, 5, 6), (x_[0], x_[1], x_[2], x_[3], x_[4], x_[5]), fontsize=12)
    plt.title('颈动脉超声样本统计', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig("Sample Statistic.svg", format="svg")
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
