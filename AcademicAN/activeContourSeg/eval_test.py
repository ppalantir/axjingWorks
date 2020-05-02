import os
import cv2
import time
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import per_image_evaluation
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib as mpl
from object_detection.utils import metrics

mpl.use("TKAgg")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
myfont = mpl.font_manager.FontProperties(
    fname='/usr/share/fonts/truetype/arphic/uming.ttc')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)


# get file_name, file_path
def get_file_name_path(dir_file, format_key):
    jpg_file_name = []
    jpg_file_path = []
    for root, dirs, files in os.walk(dir_file):
        for f in files:
            if format_key in f:
                fp = os.path.join(root, f)
                jpg_file_name.append(f)
                jpg_file_path.append(fp)
    # print("------------")
    # print(jpg_file_name)
    # print(jpg_file_path)
    # print("-------------")

    return jpg_file_name, jpg_file_path

def read_xml(f_path):

    xml_list = []
    filenames = []
    classes = []
    boxes = []
    label_s = {'plaque': 1, 'sclerosis': 2, 'vessel': 3}
    
    tree = ET.parse(f_path)
    root = tree.getroot()
    for member in root.findall('object'):
        if member[0].text != 'pseudomorphism' and member[0].text != 'normal' and member[0].text != "blood vessel":
            label = label_s.get(member[0].text)
            value = [root.find('filename').text,
                    # int(root.find('size')[0].text),
                    # int(root.find('size')[1].text),
                    label,
                    float(member[4][1].text),
                    float(member[4][0].text),
                    float(member[4][3].text),
                    float(member[4][2].text)
                    ]
            xml_list.append(value)
    return xml_list

def campute_IoU(gt_box, pre_box):
    '''
    gt_box:shape=[n,4]
    pre_box:shape=[m,4]
    '''
    gt_ymin, gt_xmin, gt_ymax,gt_xmax=gt_box
    pre_ymin, pre_xmin, pre_ymax,pre_xmax=gt_box
    
    cymin = np.maximum(gt_ymin, pre_ymin)
    cxmin = np.maximum(gt_xmin, pre_xmin)
    cymax = np.minimum(gt_ymax, pre_ymax)
    cxmax = np.minimum(gt_xmax, pre_xmax)

    cw = np.maximum(cxmax - cxmin + 1., 0.)
    ch = np.maximum(cymax - cymin + 1., 0.)
    inters = cw * ch


class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/trained-inference-graphs_blood/frozen_inference_graph.pb"
        self.PATH_TO_LABELS = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/annotations/label_map3.pbtxt"
        self.NUM_CLASSES = 3
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()
        category_index = self.category_index

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def get_detect_info(self, image_path, threshold):
        
        jpg_file_name, jpg_file_path = get_file_name_path(
            image_path, format_key=".jpg")
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                computer_label = 3
                # print(jpg_file_path)
                info_list = []
                for fp, i in zip(jpg_file_path, range(len(jpg_file_path))):

                    image = cv2.imread(fp)
                    # size = np.shape(image)
                    height, width, depth = image.shape
                    print(height, width, depth)
                    groundtruth_boxes = []
                    xml_info = read_xml(fp[:-4] + ".xml")
                    for x in range(len(xml_info)):
                        print(xml_info[x][1])
                        if int(xml_info[x][1]) == computer_label:
                            groundtruth_boxes.append(xml_info[x][2:])
                    print(groundtruth_boxes)
                    # print(size)
                    image_np_expanded = np.expand_dims(image, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name(
                        'image_tensor:0')
                    boxes = self.detection_graph.get_tensor_by_name(
                        'detection_boxes:0')
                    scores = self.detection_graph.get_tensor_by_name(
                        'detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name(
                        'detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                                    
                    
                    py_scores = np.array(scores[0])
                    py_classes = np.array(classes[0])
                    py_boxes = np.array(boxes[0])
                    scores = np.squeeze(scores)
                    classes = np.squeeze(classes)
                    groundtruth_groundtruth_is_difficult_list = np.array(np.ones(len(groundtruth_boxes)), dtype=bool)
                    groundtruth_groundtruth_is_group_of_list = np.array(np.ones(len(groundtruth_boxes)), dtype=bool)
                    #print("=====:", groundtruth_groundtruth_is_group_of_list)
                    detected_boxes = []
                    detected_scores = []
                    for num in range(len(py_scores)):
                        # print(py_scores[0])
                        #print(num)
                        if classes[num] == computer_label:
                            #print(py_boxes[num])
                            boxes = [float(py_boxes[num][0] * height), float(py_boxes[num][1] * width), float(py_boxes[num][2] * height), float(py_boxes[num][3] * width)]
                            detected_boxes.append(boxes)
                            detected_scores.append(scores[num])
                    #print(detected_boxes,"\n", ":", detected_scores)

                    print("++++++++++++++++++++++++++++++")
                    
                    num_groundtruth_classes = 3
                    matching_iou_threshold = 0.5
                    nms_iou_threshold = 1.0
                    nms_max_output_boxes = 10000
                    group_of_weight = 0.5
                    eval = per_image_evaluation.PerImageEvaluation(num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold, nms_max_output_boxes, group_of_weight)
                    scores, tp_fp_labels = eval._compute_tp_fp_for_single_class(np.array(detected_boxes),np.array(detected_scores),np.array(groundtruth_boxes),groundtruth_groundtruth_is_difficult_list, groundtruth_groundtruth_is_group_of_list)
                    print(scores, "\n", tp_fp_labels)

if __name__ == "__main__":
    # batch detecion video slice image, general label.txt
    label_map={1: "plaque", 2: "sclerosis", 3: "vessel"}
    path="/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/InceptionTrain/images/verify"
    threshold=0.5
    detecotr=TOD()
    info_list=detecotr.get_detect_info(path, threshold)            