import numpy as np 
import scipy as sp 
from scipy.optimize import leastsq
import matplotlib.pyplot as plt 

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# 加入正则项
def residuals_func_regularization(p, x, y):
    regularization = 0.0001
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p))) # L2正则化
    return ret

def fitting(M, x, y):
    '''
    M:多项式的次数
    '''
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print("Fitting Parameters:", p_lsq[0])

    return p_lsq

if __name__ == "__main__":
    

    

    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 1000)
    # 将入正态分布噪声的目标函数值
    y_ = real_func(x)
    y = [np.random.normal(0, 0.1) + y1 for y1 in y_]
    
    p_lsq_0 = fitting(0, x, y)
    p_lsq_2 = fitting(2, x, y)
    p_lsq_3 = fitting(3, x, y)
    p_lsq_9 = fitting(9, x, y)

    p_init = np.random.rand(9+1)
    #ret = residuals_func_regularization(p_init, x, y)
    p_lsq_regula = leastsq(residuals_func_regularization, p_init, args=(x, y))
    # 可视化
    plt.plot(x_points, real_func(x_points), label="real")
    plt.plot(x_points, fit_func(p_lsq_0[0], x_points), label="fitted curve 0")
    plt.plot(x_points, fit_func(p_lsq_2[0], x_points), "b", label="fitted curve 2")
    plt.plot(x_points, fit_func(p_lsq_3[0], x_points), "o", label="fitted curve 3")
    plt.plot(x_points, fit_func(p_lsq_9[0], x_points), "r", label="fitted curve 9")
    plt.plot(x_points, fit_func(p_lsq_regula[0], x_points), "y", label="fitted curve reg")
    plt.plot(x, y, "bo", label="noise")
    plt.legend()
    plt.show()