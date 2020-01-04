import os
import cv2
import time
import json
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from object_detection.utils import metrics
import matplotlib as mpl
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
# from snake import getCircleContour
from skimage.segmentation.morphsnakes import morphological_geodesic_active_contour, inverse_gaussian_gradient,circle_level_set, ellipse_level_set


mpl.use("TKAgg")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
myfont = mpl.font_manager.FontProperties(
    fname='/usr/share/fonts/truetype/arphic/uming.ttc')

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

'''读取json seg文件'''
def readjson(js_path, match_name):
    with open(js_path) as f:
        js = json.load(f)
        for k, v in js.items():
            if v["filename"]==match_name:
                gt_seg = v
        
    return gt_seg


class TOD(object):
    def __init__(self):
        #横切
        self.PATH_TO_CKPT = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/trained-inference-graphs_blood/frozen_inference_graph.pb"
        #纵切
        # self.PATH_TO_CKPT = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Longitudinal/inceptionResNetTrain/trained-inference-vessel/output_inference_graph_v1.pb/frozen_inference_graph.pb"
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
        # image = cv2.imread(image_path)
        # size = image.shape

        jpg_file_name, jpg_file_path = get_file_name_path(
            image_path, format_key=".jpg")
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

                # print(jpg_file_path)
                info_list = []
                for fp, i in zip(jpg_file_path, range(len(jpg_file_path))):

                    image = cv2.imread(fp)
                    size = np.shape(image)
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
                    # Visualization of the results of a detection.
                    # print(num_detections)
                    # print(classes.shape)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    result_list = []
                    for num in range(int(num_detections[0])):
                        py_scores = np.squeeze(scores)[num]
                        py_classes = np.squeeze(classes)[num]
                        py_boxes = np.squeeze(boxes)[num]
                        # print(py_scores[0])

                        if py_scores > threshold:
                            result_l = [fp, py_classes,
                                        py_boxes, py_scores, 1.0]
                            result_list.append(result_l)
                    info_list.append(result_list)
                    # else:
                    #     result_l = [fp, int(py_classes[num]), py_boxes[num], py_scores[num], 0.0]

                    # print('save %dst txt file: %s'%(i+ 1, txt_path))
                    np.save("./paperSegSnake/incepResNet-L_RPN-EdgeSnake.npy", info_list)
                return info_list


class TOD_L(object):
    def __init__(self):
        #纵切
        self.PATH_TO_CKPT = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Longitudinal/inceptionResNetTrain/trained-inference-vessel/output_inference_graph_v1.pb/frozen_inference_graph.pb"
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
                info_list = []
                for fp, i in zip(jpg_file_path, range(len(jpg_file_path))):
                    print(fp)

                    image = cv2.imread(fp)
                    size = np.shape(image)
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

                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    result_list = []
                    for num in range(int(num_detections[0])):
                        py_scores = np.squeeze(scores)[num]
                        py_classes = np.squeeze(classes)[num]
                        py_boxes = np.squeeze(boxes)[num]
                       
                        if py_scores > threshold:
                            result_l = [fp, py_classes,
                                        py_boxes, py_scores, 1.0]
                            result_list.append(result_l)
                    print(result_list)
                    info_list.append(result_list)

                    np.save("./paperSegSnake/incepResNet-L_RPN-EdgeSnake_L.npy", info_list)
                return info_list

def read_xml(f_path):
    xml_list = []
    filenames = []
    classes = []
    boxes = []
    label_s = {'plaque': 1, 'sclerosis': 2, 'pseudomorphism': 3, 'normal': 4}
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
    xml_list:[['./SegTest/陈胜球-右-斑块-横切.jpg', 1.0,
        array([0.48756456, 0.46503294, 0.5935851 , 0.62509483], dtype=float32), 0.99999726, 1.0]]
    return p_box = [xmin,ymin,xmax,yamx]
    '''

    for i in range(len(info_list)):
        img = cv2.imread(info_list[0][0])
        height, width, depth = img.shape
        print(height, width, depth)
        ymin, xmin, ymax, xmax = [int(info_list[0][2][0] * height), int(info_list[0][2][1] * width), int(info_list[0][2][2] * height), int(info_list[0][2][3] * width)]
        print(ymin, xmin, ymax, xmax)
    return ymin, xmin, ymax, xmax


def snake_cnn(info_list, gt_segPath, control=None):
    """
    info_list：DL模型检测信息
    gt_segPath：分割图像存储路径
    control=None：T代表横切， L代表纵切
    """
    labelMap = {1.0: "plaque", 2.0: "sclerosis", 3.0: "vessel"}
    patient_name={"蔡春灵":"患者a", "陈胜球":"患者b", "彭伟洪":"患者c", "胡瑞景":"患者d", "陈善威":"患者e", "熊伟":"患者f", "文创富":"患者g", "章克亮":"患者h"}

    im_snake_step = []
    im_snake_dist = []
    figure = plt.figure(figsize=(12,5.5))
    
    for im in range(len(info_list)):
        print(info_list[im][0][0])
        file_n_0 = info_list[im][0][0].split('/')
        file_n_ = file_n_0[-1][:-4].split("-")
        file_n = patient_name[file_n_[0]] + "-" + file_n_[1] + "-" + file_n_[2]
        img_ = cv2.imread(info_list[im][0][0])  # 读入图像
        height, width, depth = img_.shape
        img = rgb2gray(img_)  # 灰度化
        seg_img = cv.Canny(img_, 150, 200)
        lalps_img = cv.Laplacian(img_, cv.CV_16S, ksize=3)
        plt.subplot(2, 4, im + 1)
        plt.imshow(img, cmap="gray")
        
        color_dict = {"vessel":'b--',"sclerosis":"g--", "plaque":"y--"}

        im_step = []
        im_dist = []
        for i in range(len(info_list[im])):
            print(info_list[im][i][1])
            
            ymin, xmin, ymax, xmax = [int(info_list[im][i][2][0] * height), int(info_list[im][i][2][1] * width), int(info_list[im][i][2][2] * height), int(info_list[im][i][2][3] * width)]
            
            # 根据CNN预测坐标，定义椭圆方程
            x_axis_ellipse = abs(xmax-xmin)/2
            y_axis_ellipse = abs(ymin-ymax)/2
            x_centre = xmin + x_axis_ellipse
            y_centre = ymin + y_axis_ellipse
            # 圆的参数方程：(220, 100) r=100
            t = np.linspace(0, 2*np.pi, 500)  # 参数t, [0,2π]
            x = x_centre + x_axis_ellipse*np.cos(t)
            y = y_centre + y_axis_ellipse*np.sin(t)

            # 构造初始Snake
            init = np.array([x, y]).T  # shape=(400, 2)
            # init = getCircleContour((2689, 1547), (510, 380), N=200)

            # Snake模型迭代输出
            # snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.8, beta=10, gamma=0.01, w_line=-2, w_edge=500, convergence=0.1)
            # snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.8, beta=19, gamma=0.01, w_line=-2, w_edge=360, convergence=0.2)
            if control == "T":
                if float(info_list[im][i][1])==3.0:
                    snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.1, beta=10, gamma=0.01, max_iterations=1200, w_line=0, w_edge=-600, convergence=0.1)
                else:
                    snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.1, beta=30, gamma=0.01, max_iterations=1200, w_line=-1, w_edge=360, convergence=0.1)
            if control == "L":
                if float(info_list[im][i][1])==3.0:
                    # snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.2, beta=200, gamma=0.01, max_iterations=1200, w_line=20, w_edge=600, convergence=0.1)
                    continue
                else:
                    snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.1, beta=30, gamma=0.01, max_iterations=1200, w_line=-1, w_edge=360, convergence=0.1)


            im_step.append(i_l)
            im_dist.append(dist_l)
            # snake_plaque = active_contour(gaussian(img, 3), snake=init, alpha=0.8,
            #                        beta=10, gamma=0.01, w_line=1, w_edge=500, convergence=0.1)
            # 绘图显示
            print("------------:", info_list[im][i][1])
            if float(info_list[im][i][1]) == 3.0:
                color_ = "darkorchid"
            elif float(info_list[im][i][1]) == 2.0:
                color_ = "cyan"
            elif float(info_list[im][i][1]) == 1.0:
                color_ = "orangered"
            if i == 0:
                # plt.plot(init[:, 0], init[:, 1], 'r--', label="DLinit", lw=0.6)
                plt.plot(snake[:, 0], snake[:, 1], color_, label="DLeSnake", lw=0.9)
            else:
                # plt.plot(init[:, 0], init[:, 1], 'r--', lw=0.6)
                plt.plot(snake[:, 0], snake[:, 1], color_, lw=0.9)
            # plt.plot(init_plaque[:, 0], init_plaque[:, 1], '--r', lw=1)
            # plt.plot(snake_plaque[:, 0], snake_plaque[:, 1], color_p[i], lw=1)
            # font = cv2.FONT_HERSHEY_TRIPLEX
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (225, 55, 55), 4)
            # cv2.putText(img, labelMap[float(info_list[im][i][1])], (xmin+1, ymin+1), font, 1, (255, 55, 55), 1)
        
        #画出人工分割标签
        print(file_n_0[-1])
        gt_seg = readjson(gt_segPath, file_n_0[-1])
        print(file_n_0[-1], ":", gt_seg)
        lc = 0
        for gt_info in gt_seg["regions"]:
            x_seg = np.array(gt_info["shape_attributes"]["all_points_x"], dtype=int) 
            y_seg = np.array(gt_info["shape_attributes"]["all_points_y"], dtype=int) 
            label_seg = gt_info["region_attributes"]["vasc venous"]
            print(label_seg)
            if control == "T":
                if lc == 0:
                    plt.plot(x_seg, y_seg, color_dict[label_seg], label="GTSeg", lw=1.2)
                else:
                    plt.plot(x_seg, y_seg, color_dict[label_seg], lw=1.2)
            elif control == "L":
                if label_seg == "vessel":
                    continue
                else:
                    if lc == 0:
                        plt.plot(x_seg, y_seg, color_dict[label_seg], label="GTSeg", lw=1.2)
                    else:
                        plt.plot(x_seg, y_seg, color_dict[label_seg], lw=1.2)
            lc = lc + 1
        
        plt.xticks([]), 
        # plt.yticks([]),
        plt.axis("off")
        # plt.ylabel(file_n)
        # figure.legend(loc="upper right", fontsize=9,frameon=True,ncol=2,shadow=True)
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.01,hspace=0.03)
        plt.title(file_n, fontsize=9, fontproperties=myfont)
        # plt.title(file_n_0[-1][:-4], fontsize=9, fontproperties=myfont)

        im_snake_step.append(im_step)
        im_snake_dist.append(im_dist)
    if control == "T":
        plt.suptitle("IncepResNet-L_RPN-EdgeSnake Transverse", fontsize=10)
    else:
        plt.suptitle("IncepResNet-L_RPN-EdgeSnake Longitudinal", fontsize=10)

    np.save('./'+control+"_im_snake_step.npy", im_snake_step)
    np.save('./'+control+"_im_snake_dist.npy", im_snake_dist)
    plt.savefig('./'+control+'_edge_Snake.svg', format="svg")

def Morphsnake_cnn(info_list, gt_segPath, control=None):
    """
    info_list：DL模型检测信息
    gt_segPath：分割图像存储路径
    control=None：T代表横切， L代表纵切
    """
    labelMap = {1.0: "plaque", 2.0: "sclerosis", 3.0: "vessel"}
    patient_name={"蔡春灵":"患者a", "陈胜球":"患者b", "彭伟洪":"患者c", "胡瑞景":"患者d", "陈善威":"患者e", "熊伟":"患者f", "文创富":"患者g", "章克亮":"患者h"}

    im_snake_step = []
    im_snake_dist = []
    figure = plt.figure(figsize=(12,5.5))
    
    for im in range(len(info_list)):
        print(info_list[im][0][0])
        file_n_0 = info_list[im][0][0].split('/')
        file_n_ = file_n_0[-1][:-4].split("-")
        file_n = patient_name[file_n_[0]] + "-" + file_n_[1] + "-" + file_n_[2]


        Img = cv2.imread(info_list[im][0][0])  # 读入原图
        height, width, depth = Img.shape
        Image = Img
        image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        img = np.array(image, dtype=np.float64)  # 读入到np的array中，并转化浮点类型

        # 画初始轮廓
        # gimg = inverse_gaussian_gradient(image, alpha=300, sigma=4.6)
        gimg = inverse_gaussian_gradient(image, alpha=110, sigma=6)

        plt.subplot(2, 4, im + 1)
        plt.imshow(img, cmap="gray")
        
        color_dict = {"vessel":'b--',"sclerosis":"g--", "plaque":"y--"}

        im_step = []
        im_dist = []
        for i in range(len(info_list[im])):
            print(info_list[im][i][1])
            
            ymin, xmin, ymax, xmax = [int(info_list[im][i][2][0] * height), int(info_list[im][i][2][1] * width), int(info_list[im][i][2][2] * height), int(info_list[im][i][2][3] * width)]
            
            # 根据CNN预测坐标，定义椭圆方程
            x_axis_ellipse = abs(xmax-xmin)/2
            y_axis_ellipse = abs(ymin-ymax)/2
            x_centre = xmin + x_axis_ellipse
            y_centre = ymin + y_axis_ellipse
            # 构建水平集
            
            
            thresh = np.sum(image[ymin:ymax, xmin:xmax])/np.sum(image[24:1000, 18:750])
            print("短轴半径：", x_axis_ellipse)
            print("长轴半径：", y_axis_ellipse)
            print("阈值：", thresh)
            # eils = ellipse_level_set(image.shape, (y_centre, x_centre), y_axis_ellipse, x_axis_ellipse)

            # Snake模型迭代输出
            # snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.8, beta=10, gamma=0.01, w_line=-2, w_edge=500, convergence=0.1)
            # snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.8, beta=19, gamma=0.01, w_line=-2, w_edge=360, convergence=0.2)
            if control == "T":
                eils = ellipse_level_set(image.shape, (y_centre, x_centre), y_axis_ellipse, x_axis_ellipse)
                if float(info_list[im][i][1])==3.0:
                    cils = circle_level_set(image.shape, (y_centre, x_centre), min(x_axis_ellipse, y_axis_ellipse))
                    snake = morphological_geodesic_active_contour(gimg, 19, cils, threshold=thresh,smoothing=2,balloon=-1)
                    # snake_1 = morphological_geodesic_active_contour(gimg, 18, cils, threshold=thresh,smoothing=2,balloon=1)
                elif float(info_list[im][i][1])==2.0:
                    snake = morphological_geodesic_active_contour(gimg, 15, eils, threshold=thresh, smoothing=2,balloon=-1)
                    # snake_1 = morphological_geodesic_active_contour(gimg, 15, eils, threshold=thresh, smoothing=2,balloon=1)
                elif float(info_list[im][i][1])==1.0:
                    snake = morphological_geodesic_active_contour(gimg, 19, eils, threshold=thresh, smoothing=2,balloon=-1)
                    # snake_1 = morphological_geodesic_active_contour(gimg, 10, eils, threshold=thresh, smoothing=2,balloon=1)
                    
            if control == "L":
                eils = ellipse_level_set(image.shape, (y_centre, x_centre), y_axis_ellipse-5, x_axis_ellipse)
                if float(info_list[im][i][1])==3.0:
                    # snake, i_l, dist_l = active_contour(gaussian(img, 3), snake=init, alpha=0.2, beta=200, gamma=0.01, max_iterations=1200, w_line=20, w_edge=600, convergence=0.1)
                    continue
                elif float(info_list[im][i][1])==2.0:
                    # eils = ellipse_level_set(image.shape, (y_centre, x_centre), y_axis_ellipse-6, x_axis_ellipse-6)
                    snake = morphological_geodesic_active_contour(gimg, 9, eils, threshold=thresh, smoothing=2,balloon=-1)
                    # snake_1 = morphological_geodesic_active_contour(gimg, 20, eils, threshold=thresh, smoothing=2,balloon=1)
                elif float(info_list[im][i][1])==1.0:
                    snake = morphological_geodesic_active_contour(gimg, 15, eils, threshold=thresh, smoothing=2,balloon=-1)
                    # snake_1 = morphological_geodesic_active_contour(gimg, 22, eils, threshold=thresh, smoothing=2,balloon=1)
            # print("Snake：\n", snake,"Snake shape:\n", np.array(snake).shape)
            # 绘图显示
            if float(info_list[im][i][1]) == 3.0:
                color_ = "darkorchid"
            elif float(info_list[im][i][1]) == 2.0:
                color_ = "cyan"
            elif float(info_list[im][i][1]) == 1.0:
                color_ = "orangered"
            if i == 0:
                # plt.plot(init[:, 0], init[:, 1], 'r--', label="DLinit", lw=0.6)
                # plt.plot(snake, color_, label="DLeSnake", lw=0.9)
                cs = plt.contour(snake, colors=color_)
                
                # plt.contour(snake, [0.3], colors=color_)
            else:
                # plt.plot(init[:, 0], init[:, 1], 'r--', lw=0.6)
                # plt.plot(snake[:, 0], snake[:, 1], color_, lw=0.9)
                plt.contour(snake, colors=color_)
            # plt.contour(eils, colors="c")        
        
        #画出人工分割标签
        gt_seg = readjson(gt_segPath, file_n_0[-1])
        print(file_n_0[-1], ":\n", gt_seg)
        lc = 0
        for gt_info in gt_seg["regions"]:
            x_seg = np.array(gt_info["shape_attributes"]["all_points_x"], dtype=int) 
            y_seg = np.array(gt_info["shape_attributes"]["all_points_y"], dtype=int) 
            label_seg = gt_info["region_attributes"]["vasc venous"]
            # print(label_seg)
            if control == "T":
                if lc == 0:
                    plt.plot(x_seg, y_seg, color_dict[label_seg], label="GTSeg", lw=1.2)
                else:
                    plt.plot(x_seg, y_seg, color_dict[label_seg], lw=1.2)
            elif control == "L":
                if label_seg == "vessel":
                    continue
                else:
                    if lc == 0:
                        plt.plot(x_seg, y_seg, color_dict[label_seg], label="GTSeg", lw=1.2)
                    else:
                        plt.plot(x_seg, y_seg, color_dict[label_seg], lw=1.2)
            lc = lc + 1

        plt.xticks([]), 
        # plt.yticks([]),
        plt.axis("off")
        # plt.ylabel(file_n)
        # figure.legend(loc="upper right", fontsize=9,frameon=True,ncol=2,shadow=True)
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.01,hspace=0.03)
        plt.title(file_n, fontsize=9, fontproperties=myfont)
        # plt.title(file_n_0[-1][:-4], fontsize=9, fontproperties=myfont)

        im_snake_step.append(im_step)
        im_snake_dist.append(im_dist)
    if control == "T":
        plt.suptitle("IncepResNet-L_RPN-MorphSnake Transverse", fontsize=10)
    else:
        plt.suptitle("IncepResNet-L_RPN-MorphSnake Longitudinal", fontsize=10)
    plt.savefig('./'+control+'_Morph_Snake.svg', format="svg")
    


def plt_snake_dist(step_path, dist_path):
    im_snake_step = np.load(step_path)
    im_snake_dist = np.load(dist_path)
    mn = im_snake_dist.shape

    print(mn)
    print(im_snake_dist.size)
    plt.figure()
    for i in range(len(im_snake_dist)):

        for j in range(len(im_snake_dist[i])):
            # plt.subplot(7, 3, j + 1)
            plt.plot(im_snake_step[i][j], im_snake_dist[i][j])
    plt.show()


if __name__ == "__main__":

    # 参数信息
    #横切
    path = "./paperSegSnake/Use/T"
    js_path = "./paperSegSnake/via_export_json_Tv2.json"
    # 生成DL目标识别信息
    # threshold = 0.5
    # detecotr = TOD()
    # info_list = detecotr.get_detect_info(path, threshold)

    # EdgeSnake分割
    info_list = np.load("./paperSegSnake/incepResNet-L_RPN-EdgeSnake.npy")
    # snake_cnn(info_list, js_path, control="T")
    Morphsnake_cnn(info_list, js_path, control="T")

    # 能量变化图示
    # im_snake_step_path = "/home/andy/anaconda3/ANCODE/axjingWorks/workspace/AcademicAN/TwoStage/im_snake_step.npy"
    # im_snake_dist_path = "/home/andy/anaconda3/ANCODE/axjingWorks/workspace/AcademicAN/TwoStage/im_snake_dist.npy"
    # plt_snake_dist(im_snake_step_path, im_snake_dist_path)

    # 纵切
    path_L = "./paperSegSnake/Use/L"
    js_path_L = "./paperSegSnake/via_export_json_Lv3.json"
    # 生成DL目标识别信息
    # threshold = 0.5
    # detecotr_L = TOD_L()
    # info_list_L = detecotr_L.get_detect_info(path_L, threshold)
    # EdgeSnake分割
    info_list_L = np.load("./paperSegSnake/incepResNet-L_RPN-EdgeSnake_L.npy")
    # snake_cnn(info_list_L, js_path_L, control="L")
    Morphsnake_cnn(info_list_L, js_path_L, control="L")


    plt.show()

