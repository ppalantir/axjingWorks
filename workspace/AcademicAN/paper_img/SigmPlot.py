'''
@Author: your name
@Date: 2020-03-20 08:10:10
@LastEditTime: 2020-03-20 08:55:09
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\workspace\AcademicAN\paper_img\SigmPlot.py
'''
import matplotlib.pyplot as plt
import numpy as np

def elu(x,a):
    y = x.copy()
    for i in range(y.shape[0]):
        if y[i]<0:
            y[i] = a * (exp(y[i])-1)
    return y

x = np.linspace(-10,10)
y_sigmoid = 1/(1+np.exp(-x))
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

fig = plt.figure()
# plot sigmoid
ax = fig.add_subplot(221)
ax.plot(x,y_sigmoid)
ax.grid()
ax.set_title('Sigmoid')

# plot tanh
ax = fig.add_subplot(222)
ax.plot(x,y_tanh, "r")
ax.grid()
ax.set_title('Tanh')

# plot relu
ax = fig.add_subplot(223)
y_relu = np.array([0*item  if item<0 else item for item in x ]) 
ax.plot(x,y_relu, "y")
ax.grid()
ax.set_title('ReLu')

#plot leaky relu
ax = fig.add_subplot(224)
y_relu = np.array([0.2*item  if item<0 else item for item in x ]) 
ax.plot(x,y_relu, "g")
ax.grid()
ax.set_title('Leaky ReLu')

plt.tight_layout()
plt.savefig("Sigmo.png")
plt.show()