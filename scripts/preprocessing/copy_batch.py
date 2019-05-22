import os
import shutil
import numpy as np
import re 


old_path = '/home/andy/下载/2019-01-21'
xlm_path = '/home/andy/dataVOC/Annotation/'
jpg_path = '/home/andy/dataVOC/JPEGImage/'
dcm_path = '/home/andy/dataVOC/DicomFiles/'

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


def copy_batch(old_path, copy_dir):
    path_files, name_files = name_path_files(old_path)
    for pf, nf in zip(path_files, name_files):
        print(pf,'\t', nf)
        if '.xml' in pf:
            shutil.copyfile(pf, xlm_path + nf)            
            print("copy xlm文件: %s"%(nf))
        elif '.jpg' in pf:
            shutil.copyfile(pf, jpg_path + nf)            
            print("copy jpg文件: %s"%(nf))
        elif '.dcm' in pf:
            shutil.copyfile(pf, dcm_path + nf)            
            print("copy dcm文件: %s"%(nf))




if __name__ == '__main__':
    copy_batch(old_path, xlm_path)