import os
import numpy as np
import pydicom
import cv2
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree
import pandas as pd 
import math 
import random
from bs4 import BeautifulSoup

print(random.seed(1321))
print(np.random.seed(1321))

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
    print(img_info.shape)
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
    # 自适应直方图均衡化
    # Contrast Limited Adaptive Histogram Equalization
    #img_array_list = []
    #for img in img_array:
    img_array = np.uint8(cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX))
    clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8, 8))
    #print(clahe)
    #img_array_list.append(clahe.apply(img_array))
    img_array_limited_equalized = clahe.apply(img_array)


   # img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized

img_array_limited_equalized = limite_adaptive_histogram(img_info, limit=4.0)


def image_show(img_array, frame_num=0):
    # 使用PIL显示图像
    img_bitmap = Image.fromarray(img_array)
    #plt.imshow(img_bitmap)
    return img_bitmap

img_bitmap = image_show(img_array_limited_equalized, frame_num=0)
plt.imshow(img_bitmap)
plt.show()

def dcm_jpg(path_file):
    '''dicom文件转换为jpg'''
    img_info, width, height = load_dicom_file(path_file)
    cv2.imwrite(path_file[:-4] + '.jpg', img_info)

#dcm_jpg(dicom_file)



def writeVideo(filename, img_array):
    # 将切片写成视频
    frame_num, width, height = img_array.shape
    filename_output = filename.split('.')[0] + '.avi'    
    video = cv2.VideoWriter(filename_output, -1, 16, (width, height))    
    for img in img_array:
        video.write(img)
    video.release()


def load_LIDC_xml(dicom_path, xml_path, agreement_threshold=0, only_patient=None, save_nodules=False):
    pos_lines = []
    neg_lines = []
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xmls = BeautifulSoup(markup, features="xml")
    if xmls.LidcReadMessage is None:
        return None, None, None
    patient_information = load_patient_infor(dicom_file)
    patient_ids = xmls.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
    print(patient_ids)
    patient_id = patient_information['PatientID']
    print(patient_id)
    if only_patient is not None:
        if only_patient != patient_id:
            return None, None, None
    
    # itk_img = sitk.ReadImage(dicom_path)
    # img_array = sitk.GetArrayFromImage(itk_img)
    # num_z, height, width = img_array.shape
    # origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    # spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    # rescale = spacing / settings.TARGET_VOXEL_MM


    reading_sessions = xmls.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:
        print(reading_session)
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            print("nodule ID: ", nodule_id)
            rois = nodule.find_all('roi')
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue
            
            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue
            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            #z_center -= or

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue

            malignacy = nodule.characteristics.malignancy.text
            sphericiy = nodule.characteristics.sphericity.text
            margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            texture = nodule.characteristics.texture.text
            calcification = nodule.characteristics.calcification.text
            internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            subtlety = nodule.characteristics.subtlety.text

            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
            extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            pos_lines.append(line)
            extended_lines.append(extended_line)

        nonNodules = reading_session.find_all("nonNodule")
        for nonNodule in nonNodules:
            z_center = float(nonNodule.imageZposition.text)
            # z_center -= origin[2]
            # z_center /= spacing[2]
            x_center = int(nonNodule.locus.xCoord.text)
            y_center = int(nonNodule.locus.yCoord.text)
            nodule_id = nonNodule.nonNoduleID.text
            # x_center_perc = round(x_center / img_array.shape[2], 4)
            # y_center_perc = round(y_center / img_array.shape[1], 4)
            # z_center_perc = round(z_center / img_array.shape[0], 4)
            # diameter_perc = round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)
            # print("Non nodule!", z_center)
            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
            neg_lines.append(line)

    if agreement_threshold > 1:
        filtered_lines = []
        for pos_line1 in pos_lines:
            id1 = pos_line1[0]
            x1 = pos_line1[1]
            y1 = pos_line1[2]
            z1 = pos_line1[3]
            d1 = pos_line1[4]
            overlaps = 0
            for pos_line2 in pos_lines:
                id2 = pos_line2[0]
                if id1 == id2:
                    continue
                x2 = pos_line2[1]
                y2 = pos_line2[2]
                z2 = pos_line2[3]
                d2 = pos_line1[4]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < d1 or dist < d2:
                    overlaps += 1
            if overlaps >= agreement_threshold:
                filtered_lines.append(pos_line1)
            # else:
            #     print("Too few overlaps")
        pos_lines = filtered_lines

    df_annos = pd.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos_lidc.csv", index=False)
    df_neg_annos = pd.DataFrame(neg_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_neg_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_neg_lidc.csv", index=False)

    # return [patient_id, spacing[0], spacing[1], spacing[2]]
    return pos_lines, neg_lines, extended_lines
            
xml_path = r'/media/axjing/新加卷/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/069.xml'
load_LIDC_xml(xml_path=xml_path, agreement_threshold=0, only_patient=None, save_nodules=False)