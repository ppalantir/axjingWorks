import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import xml.etree.ElementTree as ET
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib as mtl 
mtl.use('TKAgg')
mtl.get_backend()
import pylab
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_FROZEN_GRAPH = '/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Longitudinal/inceptionResNetTrain/trained-inference-vessel/output_inference_graph_v1.pb/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Longitudinal/inceptionResNetTrain/annotations/label_map3.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


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

    print(path_files)
    print('#####' * 50)
    print(name_files)
    return path_files, name_files

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
            #label = label_s.get(member[0].text)
            value = [root.find('filename').text,
                    # int(root.find('size')[0].text),
                    # int(root.find('size')[1].text),
                    #label,
                    member[0].text,
                    int(member[4][1].text),
                    int(member[4][0].text),
                    int(member[4][3].text),
                    int(member[4][2].text)
                    ]
            xml_list.append(value)
    return xml_list

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = '/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/InceptionTrain/结构'
PATH_TO_TEST_IMAGES_DIR = '/home/andy/anaconda3/ANCODE/axjingWorks/workspace/AcademicAN/paper_img/raw_test_ver_Long'

#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 77) ]
path_files, name_files = name_path_files(PATH_TO_TEST_IMAGES_DIR) 
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

i = 0
for image_path in path_files:
  image = Image.open(image_path)
  image_name = image_path.split('/')[-1]
  xml_info = read_xml(image_path[:-4]+".xml")
  print(xml_info)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  print(output_dict['detection_classes'])
  # Visualization of the results of a detection.
  image_np_p=vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  print(image_np)
  #image_np_pt = tf.image.draw_bounding_boxes(image_np_p, )
  for i in range(len(xml_info)):
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.rectangle(image_np, (xml_info[i][3], xml_info[i][2]), (xml_info[i][5], xml_info[i][4]), (225, 55, 55), 4)
    cv2.putText(image_np, xml_info[i][1], (xml_info[i][3]+1, xml_info[i][2]+1), font, 1, (255, 55, 55), 1)
  cv2.imwrite('./raw_test_ver_Long/test_' + image_name[:-4] + '.png', image_np)
  # plt.figure(figsize=IMAGE_SIZE)
  # plt.imshow(image_np)
  # plt.savefig('./raw_test_verify_result/test_' + image_name[:-4] + '.png')
#plt.show()
#pylab.show()
