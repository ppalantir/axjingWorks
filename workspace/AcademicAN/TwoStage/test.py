import sys
import os
import time
import re 
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
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
        image = Image.open(PATH_TEST_IMAGE)
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
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
            category_index, use_normalized_coordinates=True, min_score_thresh=0.8, line_thickness=2)
        f_name = re.split('/',PATH_TEST_IMAGE)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.savefig(f_name[-1])
        print('Image:{}  Num: {}  scores:{}  Time: {:.3f}s'.format(PATH_TEST_IMAGE, num_detections, np.max(np.squeeze(scores)), use_time))