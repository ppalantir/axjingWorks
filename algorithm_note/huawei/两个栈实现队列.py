'''
@Author: your name
@Date: 2020-04-07 08:03:09
@LastEditTime: 2020-04-07 08:22:45
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\两个栈实现队列.py
'''
'''

入队：将元素进栈A

出队：判断栈B是否为空，如果为空，则将栈A中所有元素pop，并push进栈B，栈B出栈；

 如果不为空，栈B直接出栈。
'''
class Solution():
    def __init__(self):
        self.stack1=[]
        self.stack2=[]
    def push(self, node):
        return self.stack1.append(node)

    def pop(self):
        if self.stack2==[]:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
        return self.stack2.pop()

if __name__ == "__main__":
    S = Solution()
    print(S.push(4))
