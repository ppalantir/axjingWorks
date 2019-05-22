import os
import re
import shutil
def name_path_files(file_dir, formatkey):
    
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

if __name__ == "__main__":
    work_dir = "../workspace/training_inception_v23/images_0"
    move_dir = "../workspace/training_inception_v23/images/train"
    xml_path, xml_name = name_path_files(work_dir, '.xml')
    jpg_path, jpg_name = name_path_files(work_dir, '.jpg')
    for x_p in xml_path:
        x_ = re.split('/', x_p)
        x_n = x_[-1][:-4]
        for j_p in jpg_path:
            j_ = re.split('/', j_p)
            j_n = j_[-1][:-4]
            if x_n == j_n:
                shutil.copy2(x_p, move_dir)   
                shutil.copy2(j_p, move_dir)
