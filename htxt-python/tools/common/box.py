'''
bounding box
'''
class Box:
    def __init__(self, box_dict):
        '''
        使用read_xml函数返回的dict进行初始化
        :param box_dict: dict in the list which read_xml returns
        '''

        self.c = box_dict['class'].lower()
        self.score = box_dict['score']
        self.top_bound = self.left_top_y = self.right_top_y = int(box_dict['topBound'])
        self.lower_bound = self.left_bottom_y = self.right_bottom_y = int(box_dict['lowerBound'])
        self.left_bound = self.left_top_x = self.left_bottom_x = int(box_dict['leftBound'])
        self.right_bound = self.right_top_x = self.right_bottom_x = int(box_dict['rightBound'])

        self.left_top = (self.left_top_x, self.left_top_y)
        self.right_bottom = (self.right_bottom_x, self.right_bottom_y)
        assert self.lower_bound > self.top_bound >= 0 and self.right_bound > self.left_bound >= 0, "[ERROR] invalid boundary value"

    def get_area(self):
        area = (self.right_bound - self.left_bound) * (self.lower_bound - self.top_bound)
        assert area > 0, "[ERROR] area should be > 0"
        return area

    def area_intersect_with(self, box):
        width = min(self.right_bound, box.right_bound) - max(self.left_bound, box.left_bound)
        height = min(self.lower_bound, box.lower_bound) - max(self.top_bound, box.top_bound)
        area = width * height if width >=0 and height >= 0 else 0
        print("......... intersect area: {}".format(area))
        return area

    def area_union_with(self, box):
        area = self.get_area() + box.get_area() - self.area_intersect_with(box)
        print("......... union area: {}".format(area))
        return area

    def compute_IoU_with(self, box):
        IoU = self.area_intersect_with(box) / self.area_union_with(box)
        print("......... IoU: {}".format(IoU))
        return IoU

    def compute_d_with(self, box):
        return 0 if self.c == box.c else 1

    def compute_f_with(self, box):
        return 0 if self.compute_IoU_with(box) >= 0.6 else 1

    def union_box_with(self, box):
        box_dict = {
            'topBound': min(self.top_bound, box.top_bound),
            'lowerBound': max(self.lower_bound, box.lower_bound),
            'leftBound': min(self.left_bound, box.left_bound),
            'rightBound': max(self.right_bound, box.right_bound),
            'class': self.c,
            'score': self.score
        }
        return Box(box_dict)