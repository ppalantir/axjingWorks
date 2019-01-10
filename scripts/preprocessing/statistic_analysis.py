import os
import re
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 


jpg_path ='/home/axjing/dataVOC/JPEGImage'

def name_path_files(file_dir):
    # 文件名及文件路径列表
    path_files = []
    name_files = []
    for roots, dirs, files in os.walk(file_dir):
        for f in files:
            tmp = os.path.join(roots, f)
            #if ('.dcm' in tmp):
            path_files.append(tmp)
            name_files.append(f)

    try:
        # user enters in the filename of the csv file to convert
        # in_filename = argv[1:]
        print("files received list :" + str(path_files))
    except (IndexError, IOError) as e:
        print("invalid file detected...")
        exit(1)
    # print(path_filename)
    # print(only_filename)
    path_files_name = np.ravel(path_files)
    only_file_name = np.ravel(name_files)

    print(path_files)
    print('#####' * 30)
    print(name_files)
    print('#####' * 30)
    return path_files, name_files

def statistic_analy():
    # 统计斑块及硬化数量
    path_files, name_files = name_path_files(jpg_path)
    num_img = len(name_files)
    print('图片的数量：', num_img)
    
    plaque_list = []
    sclerosis_list = []
    normal_list = []
    for name_file in name_files:
        name_f = re.split('-', name_file)
        plaque_pattern = re.compile(r'斑块')
        sclerosis_pattern = re.compile(r'硬化')
        normal_pattern = re.compile(r'正常')

        plaque_n = plaque_pattern.findall(name_file)
        sclerosis_n = sclerosis_pattern.findall(name_file)
        normal_n = normal_pattern.findall(name_file)
        #print(name_f)
        
        plaque_list.append(plaque_n)
        sclerosis_list.append(sclerosis_n)
        normal_list.append(normal_n)
    plaque_list = [x for x in plaque_list if x != []]
    sclerosis_list = [x for x in sclerosis_list if x != []]
    normal_list= [x for x in normal_list if x != []]
    plaque_nu = len(plaque_list)
    sclerosis_nu = len(sclerosis_list)
    normal_nu = len(normal_list)
    print('斑块数量：', plaque_nu, '\t', '硬化数量：', sclerosis_nu, '\t', '正常数量：', normal_nu)
    # print(plaque_num)
    # print('====' * 50)
    # print(sclerosis_num)
    #提供汉字支持
    mpl.rcParams[u'font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    width=0.35
    x_ = ['total', 'plaque', 'sclerosis', 'normal']
    x_1 = np.arange(len(x_))
    # x = np.array(len(x_))
    # all_num = np.array(num_img)
    # plaque_num = np.array(plaque_nu)
    # sclerosis_num = np.array(sclerosis_nu)
    # normal_num = np.array(normal_nu)
    y = [num_img, plaque_nu, sclerosis_nu, normal_nu]
    y_list = np.array(y)
    y_single_list = np.array(y)/2
    #plt.rc('font', family='SimHei', size=13)
    # plt.bar(x, all_num, width=width, label=u'总计')
    # plt.bar(x+0.5, plaque_num, width=width, facecolor = 'red', label=u'斑块')
    # plt.bar(x+1.0, sclerosis_num, width=width, facecolor = 'yellowgreen', label=u'硬化')
    # plt.bar(x+1.5, normal_num, width=width, facecolor = 'green', label=u'正常')
    plt.bar(x_1, y_list, width=width, color='green')
    plt.bar(x_1+0.35, y_single_list, width =width,facecolor = 'yellowgreen',edgecolor = 'white')
    for x,y in zip(x_1,y_list):
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    for x,y in zip(x_1,y_single_list):
        plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    # plt.text(x, all_num +0.05, '%.2f' %all_num, ha='center', va= 'bottom')
    # plt.text(x+0.5, plaque_num +0.05, '%.2f' %plaque_nu, ha='center', va= 'bottom')
    # plt.text(x+1, sclerosis_num +0.05, '%.2f' %sclerosis_nu, ha='center', va= 'bottom')
    # plt.text(x+1.5, normal_num +0.05, '%.2f' %normal_num, ha='center', va= 'bottom')
    plt.xlabel('Sample classification')
    plt.ylabel('Number of samples')
    plt.xticks((0, 0+width, 1, 1+width, 2, 2+width, 3, 3+width), ('total', 'single_t', 'plaque', 'single_p', 'sclerosis', 'single_s', 'normal', 'single_n'))
    #plt.xticks((0.5,1.5,2.5,3.5), ('single_t', 'single_p', 'single_s', 'single_n'))
    plt.title('Simple Statistic')
    plt.legend()
    plt.show()
    
        

if __name__ == '__main__':
    statistic_analy()
