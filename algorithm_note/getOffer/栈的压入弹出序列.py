'''
题目描述
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序
，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
（注意：这两个序列的长度是相等的）
'''
class Solution:
    def IsPopOrder(self,pushV,popV):
        # stack存入pushV中取得的数据
        stack = []
        while popV:
            # 如果第一个元素相等，直接弹出，不用压入stack
            if pushV and popV[0] == pushV[0]:
                popV.pop(0)
                pushV.pop(0)
            # 如果stack最后一个元素与popV中的第一元素相等，将两个元素弹出
            elif stack and stack[-1] == popV[0]:
                stack.pop()
                popV.pop(0)
            # 如果pushV中有数据,压入stack
            elif pushV:
                stack.append(pushV.pop(0))
            # 以上均不满足，返回false
            else:
                return False
        return True
