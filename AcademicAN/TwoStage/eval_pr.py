import sys
import sys
import os
import time
import re 
import csv
import cv2
import glob
import tensorflow as tf
import numpy as np
import xml.dom.minidom
from xml.dom.minidom import parse
from xml.etree import ElementTree as ET
#import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import per_image_evaluation
from object_detection.utils import metrics

def read_xml(f_path):
    xml_list = []
    filenames = []
    classes = []
    boxes = []
    label_s = {'plaque':1, 'sclerosis':2, 'pseudomorphism':3, 'normal':4}
    for xml_file in glob.glob(f_path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member[0].text != 'vessel':
                label = label_s.get(member[0].text)
                value = [root.find('filename').text,
                        # int(root.find('size')[0].text),
                        # int(root.find('size')[1].text),
                        label,
                        float(member[4][0].text),
                        float(member[4][1].text),
                        float(member[4][2].text),
                        float(member[4][3].text)
                ]
                xml_list.append(value)
    return xml_list
                



if len(sys.argv) < 3:
    print('Usage: python {} test_image_path checkpoint_path'.format(sys.argv[0]))
    exit()
PATH_TEST_IMAGE = sys.argv[1]
PATH_TO_CKPT = sys.argv[2]
PATH_TO_LABELS = 'annotations/label_map.pbtxt'
NUM_CLASSES = 4
IMAGE_SIZE = (48, 32)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)
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
        start_time = time.time()
        print(time.ctime())
        #single_label = 'plaque'
        image = Image.open(PATH_TEST_IMAGE)
        
        print('---------')
        
        xml_list = read_xml('./images/verification/刘郑-右-斑块-横切.xml')
        print(xml_list, np.shape(xml_list)[0])
        gt_boxes = []
        gt_class_labels = []
        gt_is_difficult_list = []
        gt_is_group_of_list = []
        for i in range(np.shape(xml_list)[0]):
            gt_box = xml_list[i][2:]
            gt_class_label = xml_list[i][1]
            gt_boxes.append(gt_box)
            gt_class_labels.append(gt_class_label)
            gt_is_difficult_list.append(True)
            gt_is_group_of_list.append(True)
        gt_boxes = np.array(gt_boxes)
        gt_class_labels = np.array(gt_class_labels)
        gt_is_difficult_list = np.array(gt_is_difficult_list)
        gt_is_group_of_list = np.array(gt_is_group_of_list)
        print(gt_boxes, '-----', gt_class_labels)
        

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
        use_time = time.time() - start_time
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
        #     category_index, use_normalized_coordinates=True, min_score_thresh=0.8, line_thickness=2)
        eval_dicts = {'boxes':boxes, 'scores':scores, 'classes':classes, 'num_detections':num_detections}
        scores, tp_fp_labels, is_class_correctly_detected_in_image = per_image_evaluation.PerImageEvaluation().compute_object_detection_metrics(detected_boxes=np.squeeze(boxes), 
        detected_scores = np.squeeze(scores), detected_class_labels = np.squeeze(classes).astype(np.int32), groundtruth_boxes=gt_boxes, groundtruth_class_labels=gt_class_labels,groundtruth_is_difficult_list=gt_is_difficult_list, groundtruth_is_group_of_list = gt_is_group_of_list)
        #scores=np.array(scores), 
        tp_fp_labels=np.array(tp_fp_labels)
        precision, recall = metrics.compute_precision_recall(np.array(scores), tp_fp_labels[1].astype(float), 2)
        print(scores)
        print('---------')
        print(len(tp_fp_labels))
        #f_name = re.split('/',path_f)
        #print(category_index.get(value))
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        #plt.savefig(f_name[-1])
        #print('Image:{}  Num: {}  scores:{}  Time: {:.3f}s'.format(PATH_TEST_IMAGE, num_detections, np.max(np.squeeze(scores)), use_time))
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        #plt.savefig('./test_result/predicted_' + f_name[-1])
        #cv2.imwrite('./test_result/predicted_' + f_name[-1], image_np)
        