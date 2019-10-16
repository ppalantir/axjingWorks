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
        self.PATH_TO_CKPT = "./modeled/frozen_inference_graph.pb"
        self.PATH_TO_LABELS = "./annotations/label_map2.pbtxt"
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

    

# def read_img(img_path, channels):
#     img = cv2.imread(img_path,channels)
#     return img

def sobel_seg(info_list):
    img = cv2.imread(info_list[0][0])

    xmin, ymin,xmax, ymax= get_predict_box(info_list)
    #xmin = p_box[3]
    length_expanding = 0.2
    l_expanding = int((xmax - xmin) * length_expanding)
    w_expanding = int((ymax - ymin) * length_expanding)
    print(l_expanding, "JJJJJ", w_expanding)
    #edges = img[(ymin-20):ymax+20, (xmin-20):(xmax+20)]
    edges = img[(ymin-w_expanding):(ymax+w_expanding), (xmin-l_expanding):(xmax+l_expanding)]


    gray_Lapplace = cv2.Laplacian(edges, cv2.CV_16S, ksize= 3)
    dst = cv2.convertScaleAbs(gray_Lapplace)

    # plt.subplot(121)
    # plt.imshow(edges)
    # plt.subplot(122)
    # plt.imshow(dst)
    # plt.show()
    return gray_Lapplace, dst


def canny_seg(info_list, length_expanding):
    '''
        ｘml_list:  预测信息表
        length_expanding:   predict_box进行长宽缩放     float:0.5-0.8
    '''
    img = cv2.imread(info_list[0][0], 0)
    
    print(np.max(img), "img max, min", np.min(img))
    xmin, ymin,xmax, ymax= get_predict_box(info_list)
    #xmin = p_box[3]
    l_expanding = int((xmax - xmin) * length_expanding)
    w_expanding = int((ymax - ymin) * length_expanding)
    print(l_expanding, "JJJJJ", w_expanding)
    #edges = img[(ymin-20):ymax+20, (xmin-20):(xmax+20)]
    edges = img[(ymin-w_expanding):(ymax+w_expanding), (xmin-l_expanding):(xmax+l_expanding)]
    seg_edges = cv2.Canny(edges, 100, 220)
    # plt.subplot(121)
    # plt.imshow(edges)
    # plt.subplot(122)
    # plt.imshow(seg_edges)
    # plt.show()
    return edges, seg_edges

def contours_seg(info_list):
    img = cv2.imread(info_list[0][0])
    xmin, ymin,xmax, ymax= get_predict_box(info_list)
    #xmin = p_box[3]
    length_expanding = 0.2
    l_expanding = int((xmax - xmin) * length_expanding)
    w_expanding = int((ymax - ymin) * length_expanding)
    print(l_expanding, "JJJJJ", w_expanding)
    #edges = img[(ymin-20):ymax+20, (xmin-20):(xmax+20)]
    edges = img[(ymin-w_expanding):(ymax+w_expanding), (xmin-l_expanding):(xmax+l_expanding)]

    gray = cv2.cvtColor(edges,cv2.COLOR_BGR2GRAY)    
    ret, binary = cv2.threshold(gray,20,200,cv2.THRESH_BINARY)    
    print(ret, "------------:", binary)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # 输出为三个参数  
    cv2.drawContours(edges,contours,-1,(0,0,255),3)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(gray)
    # plt.subplot(122)
    # plt.imshow(edges)
    # plt.show()

    return edges
    # cv2.imshow("img", edges)    
    # cv2.waitKey(1)  
    


        

    

if __name__=="__main__":
    # batch detecion video slice image, general label.txt 
    label_map = {1:"plaque", 2:"sclerosis", 3:"pseudomorphism", 4:"normal"}
    path = "./SegTest"
    threshold = 0.5
    detecotr = TOD()
    info_list = detecotr.get_detect_info(path, threshold)

    # predict_box = info_list[0][2]
    # print(predict_box)

    #get_predict_box(info_list)

    #seg_canny(info_list, length_expanding=0.1)
    #contours_img(info_list)
    raw_img = cv2.imread(info_list[0][0])
    gray_Lapplace, dst_sobel = sobel_seg(info_list)
    edges, canny_edges = canny_seg(info_list, length_expanding=0.2)
    contours_edges = contours_seg(info_list)

    plt.subplot(321)
    plt.imshow(raw_img)
    plt.title("Raw Image")

    plt.subplot(322)
    plt.imshow(edges)
    plt.title("Faster CNN Detection Area")

    plt.subplot(323)
    plt.imshow(gray_Lapplace)
    plt.title("Laplace Edges")

    plt.subplot(324)
    plt.imshow(dst_sobel)
    plt.title("Sobel Edges")

    plt.subplot(325)
    plt.imshow(canny_edges)
    plt.title("Canny Edges")

    plt.subplot(326)
    plt.imshow(contours_edges)
    plt.title("Contours Edges")

    plt.show()


    # plaque_precision, plaque_recall, plaque_AP = get_pr_ap(info_list, 1)
    # #sclerosis_precision, sclerosis_recall, sclerosis_AP = get_pr_ap(info_list, 2)
    # #pseudomorphism_precision, pseudomorphism_recall, pseudomorphism_AP = get_pr_ap(info_list, 3)
    # #normal_precision, normal_recall, normal_AP = get_pr_ap(info_list, 4)


    # # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # plt.figure() # 创建图表1
    # plt.title('Precision/Recall threshold = 0.5')# give plot a title
    # plt.xlabel('Recall')# make axis labels
    # plt.ylabel('Precision')
    # # plt.figure(1)

    # plt.plot(plaque_recall, plaque_precision, color='green', label=label_map[1])
    # plt.annotate('AP:' + str(plaque_AP), xy=(0.55, 1), xytext=(0.45, 0.95),arrowprops=dict(facecolor='green', shrink=0.05))
    # # plt.plot(sclerosis_recall, sclerosis_precision, color='red', label=label_map[2])
    # # plt.annotate('AP:' + str(sclerosis_AP), xy=(0.15, 0.95), xytext=(0.23, 0.90),arrowprops=dict(facecolor='red', shrink=0.05))
    # # plt.plot(pseudomorphism_recall, pseudomorphism_precision, 'o-', color='skyblue', label=label_map[3])
    # # plt.annotate('AP:' + str(pseudomorphism_AP), xy=(0.13, 0.825), xytext=(0.2, 0.80),arrowprops=dict(facecolor='skyblue', shrink=0.05))
    # # plt.plot(normal_recall, normal_precision, color='blue', label=label_map[4])
    # # plt.annotate('AP:' + str(normal_AP), xy=(0.13, 0.75), xytext=(0.15, 0.70),arrowprops=dict(facecolor='blue', shrink=0.05))
    # plt.legend() # 显示图例


    # #plt.plot(recall, precision)
    # plt.savefig('p-r.png')
    # plt.show()
  
    print("Successful!!!")
