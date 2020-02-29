'''
@Author: your name
@Date: 2020-02-20 09:34:47
@LastEditTime: 2020-02-20 09:40:08
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\两个栈实现一个队列.py
'''

'''
两个栈实现一个队列
入队：元素进栈A
出队：先判断栈B是否为空，
为空则将栈A中的元素 pop 出来并 push 进栈B，再栈B出栈，如不为空则栈B直接出栈
'''
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stockA=[]
        self.stockB=[]
    def push(self, node):
        self.stockA.append(node)
    def pop(self):
        if self.stockB==[]:
            if self.stockA==[]:
                return None
            else:
                for i in range(len(self.stockA)):
                    self.stockB.append(self.stockA.pop())
        return self.stockB.pop()


'''
进栈：元素入队列A

出栈：判断如果队列A
只有一个元素，则直接出队。否则，把队A中的元素出队并入队B，
直到队A中只有一个元素，再直接出队。为了下一次继续操作，互换队A和队B

实现：就以列表作为队列的底层实现，只要保证先进先出的约束就是队列。这里只实现进栈和出栈两个操作。
'''
class Stock:
    def __init__(self):
        self.queueA=[]
        self.queueB=[]
    def push(self, node):
        self.queueA.append(node)
    def pop(self):
        if len(self.queueA)==0:
            return None
        while len(self.queueA)!=1:
            self.queueB.append(self.queueA.pop(0))
        self.queueA,self.queueB=self.queueB,self.queueA #交换是为了下一次的pop
        return self.queueB.pop()