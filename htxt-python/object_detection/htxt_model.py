import numpy as np
import os
import tensorflow as tf
from object_detection.utils import ops as utils_ops

import sys
sys.path.append("../")
from tools.common.file_sys_helper import read_image
if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util

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

def filter_output_dict(
    boxes,
    classes,
    scores,
    category_index,
    max_boxes_to_draw=20,
    min_score_thresh=.6,
    use_normalized_coordinates=True):

    detect_result_list = []

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        list_dict_item = {}
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            im_width = im_height = 500 # hard code
            if use_normalized_coordinates:
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
            else:
                (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            list_dict_item['class'] = category_index[classes[i]]['name']
            list_dict_item['score'] = round(scores[i],2)
            list_dict_item['leftBound'] = left
            list_dict_item['rightBound'] = right
            list_dict_item['topBound'] = top
            list_dict_item['lowerBound'] = bottom

            detect_result_list.append(list_dict_item)

    return detect_result_list

class HTXTModel:
    def __init__(self, path_to_model=None, num_classes=2, path_to_labels=None):
        self.PATH_TO_FROZEN_GRAPH = path_to_model + '/frozen_inference_graph.pb'
        self.NUM_CLASSES = num_classes
        self.PATH_TO_LABELS = path_to_labels
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def run_inference_for_images(self, images):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        inference_results = []
        for image in images:

            '''
            filter: ndarray of a filter (by default [500, 500, 3])
            '''
            output_dict = run_inference_for_single_image(image, detection_graph)

            inference_result = filter_output_dict(
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index)
            print(str(inference_result))
            inference_results.append(inference_result)

        return inference_results



if __name__ == '__main__':
    image_np = read_image(r'C:\Users\Serica\Workspace\htxt_python\test_set\1.tif', method='pillow')
    model = HTXTModel(path_to_model=r'C:\Users\Serica\Workspace\htxt_python\object_detection\exported_model',
                      path_to_labels=r'C:\Users\Serica\Workspace\htxt_python\object_detection\data\label_map.pbtxt')
    model.run_inference_for_images([image_np])


