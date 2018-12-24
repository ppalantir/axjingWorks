import os
import numpy as np
import pydicom
import cv2
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt

dicom_dir = r"/media/axjing/新加卷/LIDC-IDRI"
dicom_file = r"/media/axjing/新加卷/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/000001.dcm"



def name_path_files(file_dir):
    # 文件名及文件路径列表
    path_files = []
    name_files = []
    for roots, dirs, files in os.walk(file_dir):
        for f in files:
            tmp = os.path.join(roots, f)
            if ('.dcm' in tmp):
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
    print('#####' * 50)
    print(name_files)
    return path_files, name_files
#name_path_files(dicom_dir)

def load_dicom_file(dcm_path):
    read_dcm = pydicom.dcmread(dcm_path)
    img_info = read_dcm.pixel_array
    width, height = img_info.shape
    print(img_info)
    print(read_dcm)
    return img_info, width, height

img_info, width, height = load_dicom_file(dicom_file)

# def load_dicom_file_w2(file_path):
#     ds = sitk.ReadImage(file_path)
#     img_array = sitk.GetArrayFromImage(ds)
#     frame_num, width, height = img_array.shape
#     return img_array, frame_num, width, height

# img_array, frame_num, width, height = load_dicom_file_w2(dicom_file)


def load_patient_infor(dcm_file):
    # 加载病人数据存放如字典中
    patient_information = {}
    dcm_f = pydicom.read_file(dcm_file)
    patient_information['PatientID'] = dcm_f.PatientID
    patient_information['PatientName'] = dcm_f.PatientName
    patient_information['StudyInstanceUID '] = dcm_f.StudyInstanceUID 
    patient_information['SOPInstanceUID'] = dcm_f.SOPInstanceUID
    patient_information['SeriesInstanceUID '] = dcm_f.SeriesInstanceUID
    patient_information['StudyDate'] = dcm_f.StudyDate
    patient_information['StudyTime'] = dcm_f.StudyTime
    
    patient_information['Manufacturer'] = dcm_f.Manufacturer
    patient_information['Modality'] = dcm_f.Modality
    #print(dcm_f)
    #print("########" * 50)
    print(patient_information)
    return patient_information
load_patient_infor(dicom_file)

def limite_adaptive_histogram(img_array, limit=4.0):
    # Contrast Limited Adaptive Histogram Equalization
    img_array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8, 8))
        img_array_list.append(clahe.apply(img))

    img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized


def image_show(img_array, frame_num=0):
    # 使用PIL显示图像
    img_bitmap = Image.fromarray(img_array)
    #plt.imshow(img_bitmap)
    return img_bitmap

img_bitmap = image_show(img_info, frame_num=0)
plt.imshow(img_bitmap)
plt.show()

def dcm_jpg(path_file):
    img_info, width, height = load_dicom_file(dicom_file)
    cv2.imwrite(path_file[:-4] + '.jpg', img_info)

#dcm_jpg(dicom_file)

def writeVideo(img_array):
    frame_num, width, height = img_array.shape
    filename_output = filename.split('.')[0] + '.avi'    
    video = cv2.VideoWriter(filename_output, -1, 16, (width, height))    
    for img in img_array:
        video.write(img)
    video.release()

