import os
import cv2
import time
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from object_detection.utils import metrics
import matplotlib
from warnings import warn
import numpy as np
import cv2 as cv
from scipy.interpolate import RectBivariateSpline
from skimage.draw import circle_perimeter
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
#from snake import getCircleContour

matplotlib.use("TKAgg")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
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

# batch test image and general *label.txt
class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/trained-inference-graphs_blood/frozen_inference_graph.pb"
        self.PATH_TO_LABELS = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/annotations/label_map3.pbtxt"
        self.NUM_CLASSES = 2
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
        # image = cv2.imread(image_path)
        # size = image.shape
        
        jpg_file_name, jpg_file_path = get_file_name_path(image_path, format_key=".jpg")
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                
                #print(jpg_file_path)
                info_list = []
                for fp, i in zip(jpg_file_path, range(len(jpg_file_path))):
                    
                    image = cv2.imread(fp)
                    size = np.shape(image)
                    #print(size)
                    image_np_expanded = np.expand_dims(image, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    ## Visualization of the results of a detection.
                    
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
        
                    result_list=[]
                    for num in range(len(num_detections)):
                        py_scores = np.array(scores)[num]
                        py_classes = np.array(classes)[num]
                        py_boxes = np.array(boxes)[num]
                        #print(py_scores[0])
    
                        if py_scores[num] > threshold:
                            result_l = [fp, py_classes[num], py_boxes[num], py_scores[num], 1.0]
                        else:
                            result_l = [fp, int(py_classes[num]), py_boxes[num], py_scores[num], 0.0]
                    info_list.append(result_l)
                    
                    
                    #print('save %dst txt file: %s'%(i+ 1, txt_path))
                    
                print("++++++++++++++++++++++++++++++")
                print(info_list)
                return info_list
                



def get_pr_ap(info_list, num_label):
    '''
    info_list存储预测列表[文件路径，标签映射， bbox, scores, gt_label]
    num_label标签映射数字
    
    '''
    label_info = []
    num_gt_l = []
    for inf in info_list:
        if inf[1] == num_label:
            
            label_info.append(inf)
        if inf[4] == 1.0:
            num_gt_l.append(inf[4])
    num_gt = len(num_gt_l)
    label_info = np.array(label_info).T
    print(label_info)
    y_true = np.array(label_info[4][:], np.float)
    y_scores = np.array(label_info[3][:], np.float)
    
    precision, recall = metrics.compute_precision_recall(y_scores, y_true, num_gt)
    print("precision:", precision, "recall", recall)
    average_precision = metrics.compute_average_precision(precision, recall)
    average_precision = '{:.3f}'.format(average_precision)
    print("average_precision:", average_precision)
    return precision, recall, average_precision

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
            if member[0].text != 'vessel' and member[0].text != 'pseudomorphism' and member[0].text != 'normal' and member[0].text != "blood vessel":
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

def get_predict_box(info_list):
    '''
    xml_list:[['./SegTest/陈胜球-右-斑块-横切.jpg', 1.0, array([0.48756456, 0.46503294, 0.5935851 , 0.62509483], dtype=float32), 0.99999726, 1.0]]
    return p_box = [xmin,ymin,xmax,yamx]
    '''
    img = cv2.imread(info_list[0][0])
    width, height, depth = img.shape
    print(width, height, depth)
    xmin, ymin,xmax, ymax = [int(info_list[0][2][0] * height), int(info_list[0][2][1] * width), int(info_list[0][2][2] * height), int(info_list[0][2][3] * width)]
    print(xmin, ymin,xmax, ymax)
    return xmin, ymin,xmax, ymax

    
def snake_cnn(info_list):
    img_ = cv2.imread(info_list[0][0])  # 读入图像
    img = rgb2gray(img_) # 灰度化
    seg_img = cv.Canny(img_, 150, 200)
    lalps_img = cv.Laplacian(img_, cv.CV_16S, ksize= 3)

    xmin, ymin,xmax, ymax= get_predict_box(info_list)
    x_axis_ellipse = abs(xmax-xmin)/2
    y_axis_ellipse = abs(ymin-ymax)/2
    x_centre = xmin + x_axis_ellipse
    y_centre = ymin + y_axis_ellipse
    # 圆的参数方程：(220, 100) r=100
    t = np.linspace(0, 2*np.pi, 400) # 参数t, [0,2π]
    x = x_centre + x_axis_ellipse*np.cos(t)
    y = y_centre + y_axis_ellipse*np.sin(t)

    # 构造初始Snake
    init = np.array([x, y]).T # shape=(400, 2)
    #init = getCircleContour((2689, 1547), (510, 380), N=200)

    # Snake模型迭代输出
    snake = active_contour(gaussian(lalps_img,3), snake=init, alpha=0.15, beta=10, gamma=0.001, w_line=-10, w_edge=-10, convergence=2)

    # # 圆的参数方程：(220, 100) r=100
    # t = np.linspace(0, 2*np.pi, 400) # 参数t, [0,2π]
    # x_plaque = 559 + 63*np.cos(t)
    # y_plaque = 350 + 34*np.sin(t)

    # # 构造初始Snake
    # init_plaque = np.array([x_plaque, y_plaque]).T # shape=(400, 2)
    # #init = getCircleContour((2689, 1547), (510, 380), N=200)

    # # Snake模型迭代输出
    # snake_plaque = active_contour(gaussian(img,3), snake=init_plaque, alpha=0.15, beta=10, gamma=0.001, w_line=-1, w_edge=100, convergence=0.5)

    # 绘图显示
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.plot(init[:, 0], init[:, 1], '--r', lw=1)
    plt.plot(snake[:, 0], snake[:, 1], '-b', lw=1)
    #plt.plot(init_plaque[:, 0], init_plaque[:, 1], '--r', lw=1)
    #plt.plot(snake_plaque[:, 0], snake_plaque[:, 1], '-b', lw=1)
    plt.xticks([]), plt.yticks([]), plt.axis("off")

    # plt.figure()
    # plt.imshow(lalps_img)
    plt.show()


        

    

if __name__=="__main__":
    # batch detecion video slice image, general label.txt 
    label_map = {1:"plaque", 2:"sclerosis", 3:"vessel"}
    path = "./SegTest"
    threshold = 0.5
    detecotr = TOD()
    info_list = detecotr.get_detect_info(path, threshold)
    print(info_list)
    snake_cnn(info_list)

    
