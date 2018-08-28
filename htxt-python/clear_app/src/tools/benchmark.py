import os
from tools.common.box import Box
from tools.common.file_sys_helper import read_xml

class Benchmark:
    def __init__(self, path_to_grand_truth, path_to_prediction):
        '''
        init
        :param path_to_grand_truth: path to the folder contains grand truth xml files
        :param path_to_prediction: path to the folder contains prediction xml files
        '''
        self.PATH_TO_GRAND_TRUTH = path_to_grand_truth
        self.PATH_TO_PREDICTION = path_to_prediction

    def score_one_prediction(self, grand_truth_list, prediction_list):
        '''
        score one (grand truth, prediction) pair
        :param grand_truth_list: list returned from function read_xml
        :param prediction_list: list returned from function read_xml
        :return:
        '''

        '''
        e = sum(min_N(max{d(c,C),f(b,B)})
        d(c,C) = 0 if c == C otherwise 1
        f(b,B) = 0 if IoU of (b, B) >= 0.6 otherwise 1
        '''
        e = 0
        N = len(grand_truth_list)
        assert N > 0
        for prediction in prediction_list:
            box_prediction =  Box(prediction)
            score_list = []
            print("... checking for a prediction in the file:")
            for grand_truth in grand_truth_list:
                box_grand_truth = Box(grand_truth)
                d = box_grand_truth.compute_d_with(box_prediction)
                f = box_grand_truth.compute_f_with(box_prediction)
                print("...... d = {}, f = {}".format(d, f))
                score_list.append(max(d, f))
            e += min(score_list)
        return e / N

    def run_benchmark(self):
        path_to_grand_truth_xmls = [os.path.join(self.PATH_TO_GRAND_TRUTH, f) for f in os.listdir(self.PATH_TO_GRAND_TRUTH)]
        path_to_prediction_xmls = [os.path.join(self.PATH_TO_PREDICTION, f) for f in os.listdir(self.PATH_TO_GRAND_TRUTH)]
        l_pred = len(path_to_prediction_xmls)
        l_truth = len(path_to_grand_truth_xmls)
        assert l_pred == l_truth

        scores = []
        for i in range(l_pred):
            file_name = os.path.split(path_to_prediction_xmls[i])[1]
            print("Scoring file {}...".format(file_name))
            grand_truth_list = read_xml(path_to_grand_truth_xmls[i])
            prediction_list = read_xml(path_to_prediction_xmls[i])
            score = self.score_one_prediction(grand_truth_list, prediction_list)
            scores.append(score)
        print("[MESSAGE] Score of file {} is: {}".format(file_name, score))
        return score


if __name__ == '__main__':
    benchmark = Benchmark(path_to_grand_truth=r'C:\Users\Serica\Workspace\htxt_python\grand_truth',
                      path_to_prediction=r'C:\Users\Serica\Workspace\htxt_python\prediction')
    benchmark.run_benchmark()