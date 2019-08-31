import os
import numpy as np
import pydicom
import cv2
import SimpleITK as sitk

dicom_dir = r"G:\\DATA\\LIDC-IDRI"
dicom_file = r"G:\\DATA\\LIDC-IDRI\\LIDC-IDRI-0997\\01-01-2000-96481\\1491-NLST TLC VOL B30F-16103\\000257.dcm"



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
    frame_num, width, height = img_info.shape
    print(img_info)
    print(read_dcm)
    return img_info, frame_num, width, height

#load_dicom_file(dicom_file)

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
