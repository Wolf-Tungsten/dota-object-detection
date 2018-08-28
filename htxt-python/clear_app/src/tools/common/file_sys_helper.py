import xml.etree.ElementTree as ET
import math
from PIL import Image
import numpy as np
'''
和文件系统相关的工具函数集
'''
def read_xml(p):
    '''
   读取xml，返回样本标记，格式如下：
   注意标记里的坐标x y和矩阵索引x y是相反的
   [
       {
           "class":string
           "topBound"：LeftTopY里最小的
           "lowerBound"：LeftTopY里最大的
           "leftBound"：LeftTopX里最小的
           "rightBound"：LeftTopX里最大的
           "score"：概率值，如果是数据集里的label则为1
       }
       {
           ...
       }
       ...
   ]
   :param p: xml文件路径
   :return: xmlList
   '''

    xml_list = []
    xml_file_tree = ET.parse(p)
    xml_file_root = xml_file_tree.getroot()
    xml_list_item = {}
    for node in xml_file_root[4]:
        if node.tag == 'Object':
            '''都转成小写'''
            xml_list_item['class'] = node.text.lower()
        elif node.tag == 'Pixel':
            LeftTopYs = []
            LeftTopXs = []
            # 得到四个角的坐标，依次放入list中
            for coord in node:
                # 对换x y索引顺序，以符合矩阵索引
                # 对坐标下取整
                LeftTopYs.append(math.floor(float(coord.attrib['LeftTopY'])))
                LeftTopXs.append(math.floor(float(coord.attrib['LeftTopX'])))
            xml_list_item['topBound'] = min(LeftTopYs)
            xml_list_item['lowerBound'] = max(LeftTopYs)
            xml_list_item['leftBound'] = min(LeftTopXs)
            xml_list_item['rightBound'] = max(LeftTopXs)
            xml_list_item['score'] = 1 # grand truth
            xml_list.append(xml_list_item)
            xml_list_item = {}
    return xml_list

# TODO: 这个函数可用于模型最终的xml文件输出
def write_xml(direction, ETree):
    ETree.write(direction, encoding='UTF-8')

def read_image(direction, method='opencv', padding=None):
    '''
    读取一张图片，由于一开始使用的是opencv，而官方用的是pillow，故稍作改动
    :param direction: 图片路径
    :param method: opencv / pillow
    :param padding: None / stride None用于数据集清洗时，在detector中需要padding后再进行detect，此时传入detector的stride
    :return: ndarray
    '''

    img = []
    # if method == 'opencv':
    #     img = cv.imread(direction)
    if method == 'pillow':
        image = Image.open(direction)
        img = load_image_into_numpy_array(image)
    if padding is not None:
        stride = padding

        pad_horz = 0 if img.shape[0] % stride == 0 else (img.shape[0] // stride + 1) * stride - img.shape[0]
        pad_vert = 0 if img.shape[1] % stride == 0 else (img.shape[1] // stride + 1) * stride - img.shape[1]
        assert pad_horz >= 0 and pad_vert >= 0

        if pad_horz > 0 or pad_vert > 0:
            img = np.pad(img, ((0, pad_horz), (0, pad_vert), (0, 0)), 'constant')

    return img

def write_image(direction, image):
    '''
    写图片
    :param direction: 保存路径
    :param image: image ndarray
    :return: None
    '''
    pass
    # cv.imwrite(direction, image)

def load_image_into_numpy_array(image):
    '''
    官方代码的工具类 用于处理pillow读取的图片
    :param image:
    :return:
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)