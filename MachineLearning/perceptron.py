"""
使用iris数据集中两个分类的数据和[sepal length, sepal width]作为特征
进行Perceptron的学习"""
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 

def load_data():
    iris = load_iris()
    #print(iris)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    #print(df)
    df["label"] = iris.target
    #print(df["label"])
    return iris, df, df["label"]

# 数据为线性可分的二分类数据，一元一次线性方程
class PerceptronModel:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y
    
    # 随机梯度下降法
    def SGD_fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return "Perceptron Model!!!"


if __name__ == "__main__":
    iris, df, df_label = load_data()
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])

    percetorn = PerceptronModel()
    percetorn.SGD_fit(X, y)
    x_points = np.linspace(4, 7, 10)
    y_ = -(percetorn.w[0] * x_points + percetorn.b) / percetorn.w[1]

    

    plt.figure()
    plt.scatter(df[:50]["sepal length (cm)"], df[:50]["sepal width (cm)"], label="0")
    plt.scatter(df[50:100]["sepal length (cm)"], df[50:100]["sepal width (cm)"], label="1")
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.legend()

    plt.figure()
    plt.plot(x_points, y_)
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()

    plt.show()