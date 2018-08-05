import cv2 as cv
import matplotlib.pyplot as plt
import os
from tools.common.file_sys_helper import read_xml
from tools.common.file_sys_helper import read_image
from tools.common.file_sys_helper import write_image
class Splitter:
    def __init__(self, filter_shape=(500, 500), stride=100, padding=False, path_to_data_set='dataset', path_to_dist_dataset='dist_dataset',
                 threshold=0.6, verbose=True):
        '''
        init
        :param filter_shape: 图片分块大小
        :param stride: 步长
        :param padding: 是否填充
        :param path_to_data_set:
        :param path_to_dist_dataset:
        :param threshold: IoU threshold
        :param verbose: whether to print the process or not
        '''

        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.PATH_TO_DATA_SET = path_to_data_set
        self.PATH_TO_DIST_DATA_SET = path_to_dist_dataset
        self.verbose = verbose
        self.threshold = threshold

        self.image_list = []
        self.label_list = []
        '''
        初始化待处理tif与xml文件队列
        '''

        self.PATH_TO_IMAGES = [os.path.join(self.PATH_TO_DATA_SET, path)
                                for path in os.listdir(self.PATH_TO_DATA_SET)
                                if '.tif' in path]
        self.PATH_TO_LABELS = [os.path.join(self.PATH_TO_DATA_SET, path)
                                for path in os.listdir(self.PATH_TO_DATA_SET)
                                if '.xml' in path]







    def solve(self):
        '''
        主要逻辑
        :return: None
        '''

        # 一个图像文件对应一个标记文件
        assert len(self.PATH_TO_IMAGES) == len(self.PATH_TO_LABELS), "The amount of images and labels should be equal."
        # 每次迭代，分割一张图片
        self.verbose and print("Start Splitting...")
        for i in range(len(self.PATH_TO_IMAGES)):
            image = read_image(self.PATH_TO_IMAGES[i])
            labels = read_xml(self.PATH_TO_LABELS[i])

            self.verbose and print("...Splitting image: {path}".format(path=self.PATH_TO_IMAGES[i]))

            self.image_list, self.label_list = self.split(image, labels)

            self.verbose and print("Saving...")

            for j in range(len(self.image_list)):
                filename = '{}_{}.jpg'.format(i, j)
                save_path = os.path.join(self.PATH_TO_DIST_DATA_SET, filename)
                write_image(save_path, self.image_list[j])

            for j in range(len(self.label_list)):
                filename = '{}_{}.txt'.format(i, j)
                save_path = os.path.join(self.PATH_TO_DIST_DATA_SET, filename)
                file = open(save_path, 'w')
                for label in self.label_list[j]:
                    for item in label:
                        file.write(str(item) + ' ')
                    file.write('\n')
                file.close()

    def split(self, image, labels=None):
        '''
        分割一张图片
        :param image: 单张图片对应的ndarray
        :param labels: 该图片的标签对应的xml_list，若没有则仅分离图片而不处理其标签（用于detector）
        :return: image_list, label_list 用于detector时，label_list记录分割后各个图片的偏移信息
        '''

        image_list = []
        label_list = []
        # 迭代器坐标为左上角坐标，第一纬度是横向的，第二纬是纵向的
        iterator = [0, 0]
        index = 0
        while iterator[0] + self.filter_shape[0] <= image.shape[0]:
            while iterator[1] + self.filter_shape[1] <= image.shape[1]:

                top_bound = iterator[0]
                left_bound = iterator[1]
                lower_bound = iterator[0] + self.filter_shape[0]
                right_bound = iterator[1] + self.filter_shape[1]

                image_list.append(image[top_bound:lower_bound, left_bound:right_bound, :])

                if labels is not None:
                    # 对labels中的所有坐标进行偏移，随后检测有效的label
                    label_list_item = []
                    # 每次迭代，都复制一份xml label
                    _labels = labels
                    for _label in _labels:
                        _top_bound = _label['topBound']
                        _left_bound = _label['leftBound']
                        _lower_bound = _label['lowerBound']
                        _right_bound = _label['rightBound']

                        '''原标签里的box完全在图中'''
                        if _top_bound >= top_bound and _left_bound >= left_bound and _lower_bound <= lower_bound and _right_bound <= right_bound:
                            # 再把坐标换回屏幕坐标系
                            # (Class, TopLeftX, TopLeftY, LowerRightX, LowerRightY)
                            _top_left_x = _left_bound - left_bound
                            _top_left_y = _top_bound - top_bound
                            _lower_right_x = _right_bound - left_bound
                            _lower_right_y = _lower_bound - top_bound

                            assert _top_left_x >= 0 and _top_left_y >= 0 and _lower_right_x >= 0 and _lower_right_y >= 0
                            assert _top_left_x <= self.filter_shape[1] and _top_left_y <= self.filter_shape[
                                0] and _lower_right_x <= self.filter_shape[1] and _lower_right_y <= self.filter_shape[0]

                            label_list_item.append((_label['class'], _top_left_x, _top_left_y, _lower_right_x, _lower_right_y))

                            '''box部分在图中，若面积达到threshold则制作新的标签以适合'''
                        elif (_top_bound < top_bound and _lower_bound > top_bound) or (
                                _lower_bound > lower_bound and _top_bound < lower_bound) or (
                                _left_bound < left_bound and _right_bound > left_bound) or (
                                _right_bound > right_bound and _left_bound < right_bound):
                            origin_box_area = float((_lower_bound - _top_bound) * (_right_bound - _left_bound))
                            new_box_area = float((min(_lower_bound, lower_bound) - max(_top_bound, top_bound)) * (
                                        min(_right_bound, right_bound) - max(_left_bound, left_bound)))
                            if new_box_area / origin_box_area >= self.threshold:
                                _top_left_x = max(_left_bound, left_bound) - left_bound
                                _top_left_y = max(_top_bound, top_bound) - top_bound
                                _lower_right_x = min(_right_bound, right_bound) - left_bound
                                _lower_right_y = min(_lower_bound, lower_bound) - top_bound

                                assert _top_left_x >= 0 and _top_left_y >= 0 and _lower_right_x >= 0 and _lower_right_y >= 0
                                assert _top_left_x <= self.filter_shape[1] and _top_left_y <= self.filter_shape[
                                    0] and _lower_right_x <= self.filter_shape[1] and _lower_right_y <= self.filter_shape[0]

                                label_list_item.append((_label['class'], _top_left_x, _top_left_y, _lower_right_x, _lower_right_y))

                    label_list.append(label_list_item)
                else:
                    label_list_item = {}
                    label_list_item['index'] = index
                    label_list_item['leftBound'] = left_bound
                    label_list_item['rightBound'] = right_bound
                    label_list_item['topBound'] = top_bound
                    label_list_item['lowerBound'] = lower_bound
                    label_list.append(label_list_item)
                    index += 1

                iterator[1] += self.stride

            iterator[0] += self.stride
            iterator[1] = 0

        return image_list, label_list

    def label_check(self):
        '''
        检查label是否合法
        :return: None
        '''
        self.verbose and print("Start label check...")
        dist_label_paths = [self.PATH_TO_DIST_DATA_SET + path for path in os.listdir(self.PATH_TO_DIST_DATA_SET) if '.txt' in path]
        for label_file in dist_label_paths:
            with open(label_file, 'r') as f:
                self.verbose and print("... Checking label file: {path}".format(path=label_file))
                components_to_check = f.read().split(' ')
                for component in components_to_check:
                    component.strip()
                    if self.isInt(component):
                        assert int(component) >= 0, "[ERROR]: INVALID LABEL OCCURS IN FILE: {path}".format(
                            path=label_file)

    def label_image_check(self):
        '''
        检查根据label画出的box是否正确
        :return: None
        '''
        image_dirs = [os.path.join(self.PATH_TO_DIST_DATA_SET, path) for path in os.listdir(self.PATH_TO_DIST_DATA_SET) if '.jpg' in path]
        label_dirs = [os.path.join(self.PATH_TO_DIST_DATA_SET, path) for path in os.listdir(self.PATH_TO_DIST_DATA_SET) if '.txt' in path]
        for i in range(len(label_dirs)):
            # 仅当图片有标记时绘图
            if os.path.getsize(label_dirs[i]):
                labels = []
                with open(label_dirs[i]) as f:
                    for line in f:
                        line = line.strip().split()
                        labels.append(line)
                testImg = cv.imread(image_dirs[i])
                for label in labels:
                    if label[0] == 'helicopter':
                        cv.rectangle(testImg, (int(label[1]), int(label[2])), (int(label[3]), int(label[4])),
                                     (91, 192, 235), 5)
                    elif label[0] == 'plane':
                        cv.rectangle(testImg, (int(label[1]), int(label[2])), (int(label[3]), int(label[4])),
                                     (253, 231, 76), 5)
                plt.suptitle(os.path.split(image_dirs[i])[1], fontsize=20)
                plt.imshow(testImg)
                plt.show()

    def isInt(self, x):
        try:
            x = int(x)
            return isinstance(x, int)
        except ValueError:
            return False


if __name__ == '__main__':
    '''
    test script
    '''
    splitter = Splitter(path_to_data_set=r'C:\Users\Serica\Workspace\htxt_python\dataset',
                        path_to_dist_dataset=r'C:\Users\Serica\Workspace\htxt_python\dist_dataset')

    splitter.solve()
    splitter.label_image_check()