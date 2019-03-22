
import operator as opt

class KNNeighborClassification(object):
    '''
    K-nearest neighbor classification algorithm
    '''
    def __init__(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    def predict_data_list(self, test_X, K_count = 5):
        '''
            Predict test data by samples
        '''
        list_predict = []
        for index in test_X.index:
            result = self.prefict_data(test_X.iloc[index-700, :], K_count=K_count,distance_style='Manhatten')
            list_predict.append(result)
        return list_predict
    
    def prefict_data(self, test_X, distance_style = 'Euclidean', K_count = 5):
        '''
            Predict test data by one sample
        '''
        dict_data = self._get_euclidean_metric(test_X) if distance_style == 'Euclidean' else self._get_manhattan_distance(test_X)
        sort_data = dict_data.argsort()[:K_count]
        label_count = {}
        for index in sort_data:
            label = self.train_y[index]
            label_count[label] = label_count.get(label, 0) + 1
        max_label_count = sorted(label_count.items(), key = opt.itemgetter(1), reverse=True) 
        return max_label_count[0][0]

    def _get_euclidean_metric(self, test_x):
        '''
            Get Euclidean metric data
        '''
        diff_data = (test_x - self.train_X) ** 2
        sum_data = diff_data.sum(axis=1)
        dict_data = sum_data ** 0.5
        return dict_data
    
    def _get_manhattan_distance(self, test_x):
        '''
            Get Manhattan distance
        '''
        diff_data = abs(test_x - self.train_X)
        sum_data = diff_data.sum(axis = 1)
        dict_data = sum_data ** 0.5
        return dict_data
    
    def get_accuracy(self, predict_y, test_y):
        count = 0
        for index in range(len(predict_y)):
            if predict_y[index] == test_y.iloc(0)[index]:
                count += 1
        return count / len(predict_y)
