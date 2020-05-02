import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

#文件路径
#DicomFilePath = '/home/axjing/MedicalImagProcess/RawData/DicomFile/'
#JPEGFilePath = '/home/axjing/MedicalImagProcess/RawData/JPEGImagesRAW/'
DicomFilePath = '/home/axjing/MedicalImagProcess/RawData/谢燕铧181019'
JPEGFilePath = '/home/axjing/MedicalImagProcess/RawData/JPEGImages20181021/'

# 批量读取dicom文件
def batch_read_files():
    path_files_name = []
    only_file_name = []
    for roots, dirs, files in os.walk(DicomFilePath):
        for f in files:
                tmp = os.path.join(roots, f)
                if ('.dcm' in tmp):
                    path_files_name.append(tmp)
                    only_file_name.append(f)

    try:
        # user enters in the filename of the csv file to convert
        # in_filename = argv[1:]
        print("data_rnn.py received list :" + str(path_files_name))
    except (IndexError, IOError) as e:
        print("invalid file detected...")
        exit(1)
    # print(path_filename)
    # print(only_filename)
    path_files_name = np.ravel(path_files_name)
    only_file_name = np.ravel(only_file_name)

    print(path_files_name)
    print('#####' * 50)
    print(only_file_name)
    return path_files_name, only_file_name
#batch_read_files()

#.dcm文件转换为.jpg文件
def dicom_jpg(path_files, files_name):
    read_dicom_files = pydicom.dcmread(path_files, force=True)
    img_imformation = read_dicom_files.pixel_array
    print(img_imformation.shape)
    #plt.imshow(img_imformation)
    #img_imformation.save(JPEGFilePath + files_name + '.jpg')
    #plt.savefig(JPEGFilePath + files_name[:-4] + '.jpg')
    #plt.show()
    #scaled_img = cv2.convertScaleAbs(img_imformation - np.min(img_imformation),
                                     #alpha=(255.0 / min(np.max(img_imformation) - np.min(img_imformation), 10000)))
    cv2.imwrite(JPEGFilePath + files_name[:-4] + '.jpg', img_imformation)


if __name__ == '__main__':
    path_files_name, only_file_name = batch_read_files()
    for pfn, ofn in zip(path_files_name, only_file_name):
        dicom_jpg(pfn, ofn)