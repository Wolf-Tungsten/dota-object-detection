import os
from tools.image_splitter import Splitter
from tools.common import file_sys_helper
from tools.common.box import Box
from object_detection.htxtmodel import HTXTModel
import xml.etree.ElementTree as ET
import datetime
import sys

class Detector:
    def __init__(self, path_to_image_folder, path_to_output_folder, path_to_model, path_to_labels, filter_shape=(500, 500), stride=250,
                 padding=True):
        '''
        init
        :param path_to_image_folder:
        :param path_to_model:
        :param path_to_labels:
        :param filter_shape:
        :param stride:
        :param padding:
        '''
        self.PATH_TO_IMAGE_FOLDER = path_to_image_folder
        self.path_to_output_folder = path_to_output_folder
        # TODO: you can customize the file type or use reg_exp here
        self.PATH_TO_IMAGES = [os.path.join(self.PATH_TO_IMAGE_FOLDER, path)
                               for path in os.listdir(self.PATH_TO_IMAGE_FOLDER)
                               if '.jpg' in path]

        self.splitter = Splitter(filter_shape=filter_shape, stride=stride, padding=padding,
                                 path_to_data_set=self.PATH_TO_IMAGE_FOLDER, path_to_dist_dataset=None,
                                 threshold=None)

        self.model = HTXTModel(path_to_model=path_to_model,
                               path_to_labels=path_to_labels)

        self.stride = stride

    def detect_for_one_test_image(self, test_image, index=0):
        '''
        run detection for one test image, notice that the shape can be not the same as the filter_shape
        :param test_image: test image in the test set with shape [ ?, ?, 3 ]
        :return: prediction_box_list contains boxes each represents for a prediction ( the item is a Box object, see box.py)
        '''
        image_list, label_list = self.splitter.split(test_image)
        bbox_list = []
        inference_results = self.model.run_inference_for_images(image_list, batch_size=5)
        assert len(inference_results) == len(label_list)
        for (prediction_dict_list, label) in zip(inference_results, label_list):
            for prediction_dict in prediction_dict_list:
                prediction_dict['leftBound'] += label['leftBound']
                prediction_dict['topBound'] += label['topBound']
                prediction_dict['rightBound'] += label['leftBound']
                prediction_dict['lowerBound'] += label['topBound']
                bbox = Box(prediction_dict)
                bbox_list.append(bbox)
        sorted_bbox_list = sorted(bbox_list, key=lambda box: box.score)

        prediction_box_list = self.non_max_supress(sorted_bbox_list)

        return prediction_box_list

    def non_max_supress(self, bbox_list):
        flag = True
        while flag:
            flag = False
            for i in range(len(bbox_list) - 1):
                for j in range(i + 1, len(bbox_list)):
                    if not bbox_list[i].to_delete and not bbox_list[j].to_delete and bbox_list[i].compute_IoU_with(
                            bbox_list[j]) > 0.5:
                        flag = True
                        if bbox_list[i].score > bbox_list[j].score:
                            bbox_list[j].to_delete = True
                        else:
                            bbox_list[i].to_delete = True

        prediction_list = []
        for item in bbox_list:
            if not item.to_delete:
                prediction_list.append(item)

        return sorted(prediction_list, key=lambda box: (box.lower_bound, box.top_bound))

    def build_xml_output(self, box_list, name):
        root = ET.Element("Research", {"Direction": "高分软件大赛"})
        root.attrib["ImageName"] = name + ".tif"

        department = ET.SubElement(root, "Department")
        department.text = "红茶玛奇朵五分甜"

        date = ET.SubElement(root, "Date")
        date.text = datetime.datetime.now().strftime("%Y-%m-%d")

        plugin_name = ET.SubElement(root, "PluginName")
        plugin_name.text = "目标识别"

        plugin_class = ET.SubElement(root, "PluginClass")
        plugin_class.text = "检测"

        results = ET.SubElement(root, "Results", {"Coordinate": "Pixel"})
        for box in box_list:
            box.append_to_element(results)

        file_sys_helper.write_xml(
            os.path.join(self.path_to_output_folder, name+".xml"),
            ET.ElementTree(root))

    def run_detection(self):
        '''
        run detection for a test set, can handel multiple images in the test set
        :return: result_list contains results for every image in the test set
        '''
        result_list = []
        for index, path in enumerate(self.PATH_TO_IMAGES):
            image = file_sys_helper.read_image(path, method='pillow', padding=self.stride)
            result_list_item = self.detect_for_one_test_image(image, index=index)
            result_list.append(result_list_item)
            self.build_xml_output(result_list_item, os.path.basename(path).split('.')[0])


        return result_list

if __name__ == '__main__':
    '''
    test scripts
    '''
    detector = Detector(path_to_image_folder=sys.argv[1],
                        path_to_output_folder=sys.argv[2],
                        path_to_model='.\\object_detection.\\exported_model',
                        path_to_labels='.\\object_detection.\\data.\\label_map.pbtxt')

    result_list = detector.run_detection()

