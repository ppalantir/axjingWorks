import os.path as osp
import cv2
import numpy as np 
import xml.etree.ElementTree as ET


class USDataSet(object):
    # def __init__(self, trainset_dir, testset_dir):
    #     self.trainset_dir = trainset_dir
    #     self.testset_dir = testset_dir

    def name_path_files(self, file_dir, formatkey):
        '''
        file_dir 工作文件夹路径
        formatkey 要遍历文件的格式

        return
        要遍历文件路径列表、及文件名列表
        '''
        self.file_dir = file_dir
        self.formatkey = formatkey
        path_files = []
        name_files = []
        for roots, dirs, files in os.walk(file_dir):
            for f in files:
                tmp = os.path.join(roots, f)
                if formatkey in tmp:
                    path_files.append(tmp)
                    name_files.append(f)

        print(path_files)
        print('---------' * 10)
        print(name_files)
        print('---------' * 10)
        return path_files, name_files

    def read_xml(self, xml_file):
        '''
        xml_file *.xml文件路径

        return 
        [('陈胜球-左-斑块-横切.jpg', 1024, 768, 'pseudomorphism', 431, 330, 497, 415), ('陈胜球-左-斑块-横切.jpg', 1024, 768, 'plaque', 468, 244, 641, 325), ('陈胜球-左-斑块-横切.jpg', 1024, 768, 'vessel', 378, 212, 738, 521)]
        '''
        xml_list = []
        self.xml_file = xml_file
        #for xml_file in glob.glob(self.xml_path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            #if member[0].text != 'vessel' and member[0].text != 'blood vessel' and member[0].text != 'pseudomorphism' and member[0].text != 'normal':
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(float(member[4][0].text)),
                    int(float(member[4][1].text)),
                    int(float(member[4][2].text)),
                    int(float(member[4][3].text))
                    )
            xml_list.append(value)
                
        return xml_list
    
    def read_img(self, img_file):
        self.img_file = img_file
        img = cv2.imread(img_file)
        return img

if __name__ == "__main__":
    xml_file = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/ResNetTrain/陈胜球-左-斑块-横切.xml"
    USD = USDataSet()
    value = USD.read_xml(xml_file)
    print(value)
    