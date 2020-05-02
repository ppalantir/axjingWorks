import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import xml.etree.ElementTree as ET
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib as mpl
mpl.use('TKAgg')
mpl.get_backend()
import pylab
from matplotlib import pyplot as plt
from PIL import Image


mpl.use("TKAgg")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
myfont = mpl.font_manager.FontProperties(
    fname='/usr/share/fonts/truetype/arphic/uming.ttc')

def name_path_files(file_dir):
    # 文件名及文件路径列表
    path_files = []
    name_files = []
    for roots, dirs, files in os.walk(file_dir):
        for f in files:
            tmp = os.path.join(roots, f)
            if ('.xml' in tmp):
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

    # print(path_files)
    # print('#####' * 50)
    # print(name_files)
    return path_files, name_files

def read_xml(f_path):

    xml_list = []
    filenames = []
    classes = []
    boxes = []
    label_s = {'plaque': 1, 'sclerosis': 2, 'vessel': 3}
    
    tree = ET.parse(f_path)
    root = tree.getroot()
    for member in root.findall('object'):
        if member[0].text != 'vessel' and member[0].text != "blood vessel":
            #label = label_s.get(member[0].text)
            value = [root.find('filename').text,
                    # int(root.find('size')[0].text),
                    # int(root.find('size')[1].text),
                    #label,
                    member[0].text,
                    int(member[4][1].text),
                    int(member[4][0].text),
                    int(member[4][3].text),
                    int(member[4][2].text)
                    ]
            xml_list.append(value)
    return xml_list


path_files, name_files = name_path_files("/home/andy/anaconda3/ANCODE/axjingWorks/DATA/medical_img/dataVOC/Annotation/hengqie") 
color_class = {'plaque': (225, 0, 0), 'sclerosis': (225, 225, 0), 'vessel': (225, 0, 225)}
i = 0
img = cv2.imread("./USimg_range/陈满有-右-斑块-横切.jpg")
for image_path in path_files:
    
    xml_info = read_xml(image_path[:-4]+".xml")
    #print(xml_info)
    #print(img)
    i=i+1
    print(i)
    for i in range(len(xml_info)):
        font = cv2.FONT_HERSHEY_TRIPLEX
        #print(color_class.get(xml_info[i][1]))
        cv2.rectangle(img, (xml_info[i][3], xml_info[i][2]), (xml_info[i][5], xml_info[i][4]), color_class.get(xml_info[i][1]), 1)
        #cv2.putText(img, xml_info[i][1], (xml_info[i][3]-1, xml_info[i][2]-1), font, 1, (255, 55, 55), 1)
#cv2.imwrite("./USimg_range/陈满有-右-斑块-横切.png", img)

path_files_z, name_files_z = name_path_files("/home/andy/anaconda3/ANCODE/axjingWorks/DATA/medical_img/dataVOC/Annotation/zhongqie") 
img_z = cv2.imread("./USimg_range/陈满有-右-斑块-纵切.jpg")
for image_path in path_files_z:
    xml_info = read_xml(image_path[:-4]+".xml")
    #print(xml_info)
    #print(img)
    i=i+1
    print(i)
    for i in range(len(xml_info)):
        font = cv2.FONT_HERSHEY_TRIPLEX
        #print(color_class.get(xml_info[i][1]))
        cv2.rectangle(img_z, (xml_info[i][3], xml_info[i][2]), (xml_info[i][5], xml_info[i][4]), color_class.get(xml_info[i][1]), 1)
        #cv2.rectangle(img_z, (xml_info[i][3], xml_info[i][2]), (xml_info[i][5], xml_info[i][4]), (255, 0, 0), 1)
        #cv2.putText(img, xml_info[i][1], (xml_info[i][3]-1, xml_info[i][2]-1), font, 1, (255, 55, 55), 1)
#cv2.imwrite("./USimg_range/陈满有-右-斑块-纵切.png", img_z)
# cv2.imshow("Image", img)
# cv2.imshow("Image", img_z)
# cv2.waitKey(0)

plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.title("Transverse")
plt.subplot(122)
plt.imshow(img_z)
plt.title("Longitudinal")
plt.savefig("./USimg_range/rang.png")
plt.show()
  
