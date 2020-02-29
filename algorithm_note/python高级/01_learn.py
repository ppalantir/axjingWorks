'''
@Author: your name
@Date: 2020-01-15 20:49:27
@LastEditTime : 2020-01-30 21:06:31
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\01_learn.py
'''
import sys 
from imp import *

# 链表和顺序表统称为线性表；链表包括数据区和链接区；顺序表包括表头max,num,数据区
# python单链表

# python双向链表

# 网络协议TCP/IP协议族
# 链路层--》网络层--》传输层--》应用层
# 物理层--》数据链路层--》网路层--》传输层--》会话层--》表示层--》应用层
# 端口：多台电脑之间用来区分进程：区分电脑中的那一个程序
# ip地址：用来标记唯一的一台电脑，相当于身份证：区分哪一台电脑
# ip地址分为A/B/C/D/E五类

# socket简介：对个电脑间通信的方式，套接字
# 1 本地进程通信IPC:队列、同步（互斥锁、条件变量）
import socket

# UDP协议相当于写信 ， TCP相当于打电话
# 创建tcp socket (tcp套接字)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(s)
# 准备接收方地址
sendAdders = ("172.31.102.22", 8080)

# 键盘发送数据
sendData = input("输入参数：")

# 发送数据到指定电脑
s.sendto(sendData, sendAdders)

s.close()

# 创建udp socket(套接字)
# 注意：使用UDP需要每次都要写ip地址及端口
# 在同一个OS中，端口不允许相同
# UDP绑定信息
# 全双工：可说可听。半双工：可说，同时只能听，单工：同时只能说或者听
su = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
binAddr = ("本机ip", 8080)
su.bind(binAddr)

print(su)

# 闭包 
def test1(number):
    print('外部函数')
    # 在函数内部再定义一个函数，并且这个函数用到了外边函数的变量
    def test1_in(number_in):
        print("内部函数")
        return number+number_in
    return test1_in

print(test1(1))
t_in = test1(21)
print(t_in(2))
# 迭代器：所有可迭代的对象，next（a）,iter()



# 生成器:占用内存少，用的时候再生成
a = [x*2 for x in range(10)]
# 生成器第一种创建方式
b = (x*2 for x in range(10))
print(a)
print(b, next(b))
# 生成器应用：斐波那契数列
def Fei_craetcall():
    a, b = 0, 1
    for i in range(5):
        yield b
        print(b)
        a, b = b, a+b

a = Fei_craetcall()
print(next(a))


# 私有化
class propety_test():
    def __init__(self):
        self.__num = 100 # 私有方法在class不可修改

    # 函数外修改私有方法
    def setNum(self, newNum):
        self.__num = newNum
    
    def getNum(self):
        return self.__num
    
    newNum = property(getNum, setNum)
    
    @property
    def newNums(self):
        self.__num

pT = propety_test()
# print(pT.__num)

print(pT.setNum(400))
print(pT.getNum())
pT.newNum = 500
print(pT.newNum)

# pT.__num = 200 # 错误

# 位运算
# 左移完成乘法 *2
# 右移完成除法 /2
a1 = 8
print(~a1)

#------ 深拷贝和浅拷贝 ------#
# 浅拷贝：对于一个对象的顶层拷贝，指的是拷贝引用，而没有拷贝内容
a = [1, 2, 3]
b = a
print(id(a), "\n", id(b))

# 深拷贝
import copy
c = copy.deepcopy(a)
print(id(a), "\n", id(c))
q = copy.copy(a) # 创建新的空间并可复制
a.append(44)
print(id(a), "\n", id(b), "\n", id(c))
print(a, "\t", b, "\t", c)

# == 和 is 的区别？
# == 用于判断内容是否相同，而is用于判断内存地址是否指向同一个位置，判断是不是同一个


reload(sys) #重新加载
print(sys.path)
