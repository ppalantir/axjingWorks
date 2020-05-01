'''
题目描述
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
注意：保证测试中不会当栈为空的时候，对栈调用pop()或者min()或者top()方法。
思路：
- 利用辅助栈来存放最小值
1. 每入栈一次，就与辅助栈顶比较大小
2. 如果小则入栈，如果大就入栈当前辅助栈顶
3. 当出栈时，辅助栈也要出栈，确保辅助栈顶一定是当前栈的最小值
'''
class Solution:
    def __init__(self):
        self.stack = []
        self.assist = []
    def push(self,node):
        min = self.min()
        if not min or node < min:
            self.assist.append(node)
        else:
            self.assist.append(min)
        self.stack.append(node)

    def pop(self):
        if self.stack:
            self.assist.pop()
            return self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]
    def min(self):
        if self.assist:
            return self.assist[-1]