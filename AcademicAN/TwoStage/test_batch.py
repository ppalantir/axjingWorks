import sys
import os
import time
import re 
import csv
import cv2
import tensorflow as tf
import numpy as np
#import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# if len(sys.argv) < 0:
#     print('Usage: python {} test_image_path checkpoint_path'.format(sys.argv[0]))
#     exit()

def name_path_files(file_dir):
    # 文件名及文件路径列表
    path_files = []
    name_files = []
    for roots, dirs, files in os.walk(file_dir):
        for f in files:
            tmp = os.path.join(roots, f)
            if ('.jpg' in tmp):
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
    # print('#####' * 10)
    # print(name_files)
    return path_files, name_files


PATH_TO_CKPT = sys.argv[1]
PATH_TO_LABELS = 'annotations/label_map.pbtxt'
NUM_CLASSES = 4
IMAGE_SIZE = (48, 32)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        path_files, name_files = name_path_files('./images/verification/')
        writer_lists = []
        for path_f in path_files:
            start_time = time.time()
            print(time.ctime())
            image = Image.open(path_f)
            image_np = np.array(image).astype(np.uint8)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # print(classes)
            # print(num_detections)
            eval_dicts = {'boxes':boxes, 'scores':scores, 'classes':classes, 'num_detections':num_detections}
            use_time = time.time() - start_time
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                category_index, use_normalized_coordinates=True, min_score_thresh=0.5, line_thickness=2)
            #vis_util.VisualizeSingleFrameDetections.images_from_evaluation_dict(image_np,eval_dict=eval_dicts)
            #categories_glob = []
            print(category_index)
            f_name = re.split('/',path_f)
            #print(category_index.get(value))
            for index, value in enumerate(classes[0]):
                if scores[0, index] > 0.5:
                    score = scores[0, index]
                    categories_glob = category_index.get(value)
                    writer_list = [f_name[-1], categories_glob['id'], categories_glob['name'], score, use_time]
                    writer_lists.append(writer_list)
                    # print(writer_list)
                    # print(index, '---', categories_glob,'---', score )
                    print(boxes)
            
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            #plt.savefig('./test_result/predicted_' + f_name[-1])
            cv2.imwrite('./test_result/predicted_' + f_name[-1] + ".jpg", image_np)
            
            #writer_lists.append(writer_list)
            #print('Image:{}  Num: {}  classes:{}  scores:{}  Time: {:.3f}s'.format(f_name[-1], num_detections, 'np.squeeze(classes[:2])', np.max(np.squeeze(scores)), use_time))
        with open('./test_result/test_result.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['test file', 'id', 'classes', 'scores', 'time/s'])
            writer.writerows(writer_lists)

            
            
