
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from KNNAlgoritym.KNNeigboClasssification import KNNeighborClassification
from Preprocessing.DataPreprocessing import Preprocessing

# from KNNAlgoritym.KNNeigboClasssification import KNNeighborClassification
# from Preprocessing.DataPreprocessing import Preprocessing
import pandas as pd

file_path = './MachineLearning/Dataset/KNN/KNNDataSet1.txt'
feature_names = ['feature1', 'feature2', 'feature3', 'result']

def test_KNN():
    dataset = pd.read_csv(file_path, sep='\t', header=None, names=feature_names)
    train_X, train_y, test_X, test_y = Preprocessing.split_and_normalize_data(dataset, is_normalized=True)
    knn_classification = KNNeighborClassification(train_X, train_y)
    predicted_data = knn_classification.predict_data_list(test_X)
    accuracy = knn_classification.get_accuracy(predicted_data, test_y)
    print(accuracy)

def test():
    print("test")

test_KNN()