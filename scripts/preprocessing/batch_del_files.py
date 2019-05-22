import os

#批量删除有标记文件
def batch_del_files(path):
    path_files_name = []
    only_file_name = []
    for roots, dirs, files in os.walk(path):
        for f in files:
                file_path = os.path.join(roots, f)
                if '有标记' in file_path:
                    path_files_name.append(file_path)
                    os.remove(file_path)
                    print('删除有标记文件： ', file_path)
                    only_file_name.append(f)

if __name__ == '__main__':
    batch_del_files(path='/home/axjing/MedicalImagProcess/dataVOC/JPEGImages/')