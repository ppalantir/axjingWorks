import os
import cv2
import time
import glob
import pickle
import logging
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib as mpl
from object_detection.utils import metrics


logger = logging.getLogger(__name__)

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
config.gpu_options.per_process_gpu_memory_fraction = 0.99
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
    for xml_file in glob.glob(f_path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member[0].text != 'pseudomorphism' and member[0].text != 'normal' and member[0].text != "blood vessel":
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


def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    label_s = {'plaque': 1, 'sclerosis': 2, 'vessel': 3}
    objects = []
    for obj in tree.findall('object'):
        if obj[0].text != 'pseudomorphism' and obj[0].text != 'normal' and obj[0].text != "blood vessel":
            #label = label_s.get(obj[0].text)
            obj_struct = {}
            # obj_struct['name'] = label_s.get(obj.find('name').text)
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

    return objects


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
                    height, width, depth = np.shape(image)
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
                    
                    py_score = np.array(scores)[0]
                    py_classe = np.array(classes)[0]
                    py_boxe = np.array(boxes)[0]
                    py_scores = []
                    py_classes = []
                    py_boxes = []
                    for i in range(len(py_boxe)):
                        if py_score[i] > threshold:
                            ymin, xmin, ymax, xmax = [float(py_boxe[i][0] * height), float(py_boxe[i][1] * width), float(py_boxe[i][2] * height), float(py_boxe[i][3] * width)]
                            py_boxe_ = [ymin, xmin, ymax, xmax]
                            print(py_boxe_)
                            
                            py_scores.append(py_score[i])
                            py_classes.append(py_classe[i])
                            py_boxes.append(py_boxe_)       
                    single_info = [fp, py_classes, py_scores,py_boxes]
                    info_list.append(single_info)
                    print(fp,"\n","-----")
                    print(py_scores,"\n","-----")
                    print(py_classes,"\n","-----")
                    print(py_boxes,"\n","-----")
                np.save(image_path+"/detection_info.npy", info_list)

                return info_list

# F-score
def F_sorce(recall, precision, beta):
    f_sorce = ((1+beta**2)*(precision*recall))/(beta*beta*precision+recall)
    return f_sorce

# 第一种AP计算方式
def voc_eval(det_path,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(annopath):
        os.mkdir(annopath)
    cachefile = os.path.join(annopath,"GTinfo" + '_annots.pkl')
    # read list of images
    # imagesetfile 需要测试的图片集合 test.txt
    # with open(imagesetfile, 'r') as f:
    #     lines = f.readlines()
    # imagenames = [x.strip() for x in lines]
    imagenames, jpg_file_path = get_file_name_path(imagesetfile, ".jpg")
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            # parse_rec 返回每张图片的全部物品数量
            recs[imagename] = parse_rec(
                os.path.join(annopath, imagename[:-4] + '.xml'))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
            print(recs)

    # extract gt objects for this class
    # class_recs 每个图片中真实物品数, 里面的量bbox,difficult都是列表,
    # 提取所有测试图片中当前类别所对应的所有ground_truth
    class_recs = {}
    npos = 0
    for imagename in imagenames:

        # 找出所有当前类别对应的object
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # 该图片中该类别对应的所有bbox
        bbox = np.array([x['bbox'] for x in R])
        # 修改difficult 为 bool 类型数据
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # 该图片中该类别对应的所有bbox的是否已被匹配的标志位
        det = [False] * len(R)
        # 累计所有图片中的该类别目标的总数，不算diffcult
        npos = npos + sum(~difficult)
        
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    #print(class_recs)

    # read dets
    label_map = {1.0: "plaque", 2.0: "sclerosis", 3.0: "vessel"}
    detection_info = np.load(det_path)
    image_ids = []
    confidence = []
    BB = []
    for i in range(len(detection_info)):
        for j in range(len(detection_info[i][1])):
            
            if label_map[detection_info[i][1][j]] == classname:
                image_ids.append(detection_info[i][0])
                confidence.append(detection_info[i][2][j])
                BB.append(detection_info[i][3][j])
    #print(BB)
    # sort by confidence
    # argsort函数返回的是数组值从小到大的索引值
    # 将该类别的检测结果按照置信度大小降序排列
    
    # go down dets and mark TPs and FPs
    # image_ids 检测出来的数量
    # 该类别检测结果的总数（所有检测出的bbox的数目）
    nd = len(image_ids)
    # 用于标记每个检测结果是tp还是fp
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # 取出该条检测结果所属图片中的所有ground truth
        im_id = image_ids[d].split('/')[-1]
        R = class_recs[im_id]
        bb = BB[d]  # BB是检测出来的 物品坐标.
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)  # BBGT 图片真实位置坐标, 是列表
        # 计算与该图片中所有ground truth的最大重叠度
        if BBGT.size > 0:
            # compute overlaps
            # intersection  np.maximum X 与 Y 逐位比较取其大者,得到交集的两个坐标.
            ixmin = np.maximum(BBGT[:, 1], bb[0])
            iymin = np.maximum(BBGT[:, 0], bb[1])
            ixmax = np.minimum(BBGT[:, 3], bb[2])
            iymax = np.minimum(BBGT[:, 2], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)  # 求序列的最值
            jmax = np.argmax(overlaps)  # argmax返回的是最大数的索引
            print(ixmin, ixmax)
        # 如果最大的重叠度大于一定的阈值
        if ovmax > ovthresh:
            # 如果最大重叠度对应的ground truth为difficult就忽略
            if not R['difficult'][jmax]:
                # 如果对应的最大重叠度的ground truth以前没被匹配过则匹配成功，即tp
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        # 该图片中没有对应类别的目标ground truth或者与所有ground truth重叠度都小于阈值
        else:
            fp[d] = 1.
        ## print('{}/{}  ID: {} tp:{:1.0f} fp:{:1.0f}'.format(d, nd, image_ids[d], tp[d], fp[d]))

    # compute precision recall
    # 按置信度取不同数量检测结果时的累计fp和tp
    # np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # 召回率为占所有真实目标数量的比例，非减的，注意npos本身就排除了difficult，因此npos=tp+fn
    gt = float(npos)
    rec = tp / gt
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    # 精度为取的所有检测结果中tp的比例
    tp_fp = np.maximum(tp + fp, np.finfo(np.float64).eps)
    prec = tp / tp_fp
    print(prec)
    print(rec)
    error_rate = fp / tp_fp
    f1_sorce = F_sorce(rec, prec, 1)
    f2_sorce = F_sorce(rec, prec, 2)
    # 计算recall-precise曲线下面积（严格来说并不是面积）
    ap = voc_ap(rec, prec, use_07_metric)
    print('测试数据集, 图片数量: {}张  gt_Label_num {} '.format(len(imagenames), gt))
    print(
        '人工标记gt   {:.0f} \n'
        '准确识别TP   {:.0f} \n'
        '误报FP       {:.0f} \n'
        '总共检出tpfp {:.0f} \n'
        '漏检FN       {:.0f}'
        .format(gt, tp[-1], fp[-1], tp_fp[-1], gt - tp[-1]))
    print(
        '误报率fp/tp_fp        {:.4f}\n'
        '正确率prec(tp/tp_fp)  {:.4f} \n'
        '查全率rec(tp/gt)      {:.4f} \n'
        'ap  {:.4f} \n'
        'F1-sorce     {:.0f} \n'
        'F2-sorce     {:.0f}'
        .format(error_rate[-1], prec[-1], rec[-1], ap, f1_sorce[-1], f2_sorce[-1]))
    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        # 2010年以前按recall等间隔取11个不同点处的精度值做平均(0., 0.1, 0.2, …, 0.9, 1.0)
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                # 取最大值等价于2010以后先计算包络线的操作，保证precise非减
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        # 2010年以后取所有不同的recall对应的点处的精度值做平均
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        # 计算包络线，从后往前取最大保证precise非减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        # 找出所有检测结果中recall不同的点
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        # 用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# 第二种AP计算方式
def compute_ap(gt_boxes, gt_class_ids,
               pred_boxes, pred_class_ids, pred_scores,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match = np.zeros([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


if __name__ == "__main__":
    label_map = {1: "plaque", 2: "sclerosis", 3: "vessel"}
    eval_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/InceptionTrain/images/raw_test"
    threshold=0.5
    detecotr=TOD()
    info_list=detecotr.get_detect_info(eval_path, threshold)

    det_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/InceptionTrain/images/raw_test/detection_info.npy"
    annopath = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/InceptionTrain/images/raw_test/xml"
    imagesetfile = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/InceptionTrain/raw_test"
    classname = "plaque"
    # cachedir = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/InceptionTrain/images/balance_test/xml"
    rec, prec, ap = voc_eval(det_path,
             annopath,
             imagesetfile,
             classname)
    plt.figure()
    plt.plot(rec, prec)
    plt.show()
