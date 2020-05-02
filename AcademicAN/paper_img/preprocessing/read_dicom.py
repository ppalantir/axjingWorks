import os
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np 
import cv2


# 读取dicom文件
def read_dicom(file_path):
    read_dcm = pydicom.dcmread(file_path)
    print(read_dcm)

if __name__ == '__main__':
    file_path = "/home/axjing/LIDC-IDRI-0997/01-01-2000-96481/1491-NLST TLC VOL B30F-16103/000257.dcm"
    read_dicom(file_path)