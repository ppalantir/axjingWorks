import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]

class NaiveBayes:
    def __init__(self):
        self.model = None
    
    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差/方差
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X])) / float(len(X))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean, 2) / (2*math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train
    def sumarize(self, train_data):
        sumarizes = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        print("Sumarizes:", sumarizes)
        return sumarizes

    # 分别求出数学期望和方差
    def Bayers_fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label:self.sumarize(value)
            for label, value in data.items()
        }
        print("Model Info:",self.model)
        return "GaussianNaiveBayes train done!"

    # 计算概率
    def calculate_probabilitise(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev
                )
        
        return probabilities
    
    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(
            self.calculate_probabilitise(X_test).items(),
            key=lambda x: x[-1])[-1][0]
     
        print(sorted(
            self.calculate_probabilitise(X_test).items(),
            key=lambda x: x[-1]))
        return label

    # acc正确率
    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        
        return right / float(len(X_test))
    

if __name__ == "__main__":
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = NaiveBayes()
    print(model.Bayers_fit(X_train, y_train))
    score = model.score(X_test, y_test)
    print(score)
