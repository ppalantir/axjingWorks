'''
@Author: your name
@Date: 2020-03-25 17:20:52
@LastEditTime: 2020-03-25 17:24:11
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\合并两个排序的链表.py
'''
'''输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。'''

# 非递归
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        #初始化
        tmp = ListNode(0)
        pHead = tmp        
        while pHead1 and pHead2:
            if pHead1.val < pHead2.val:
                tmp.next = pHead1
                pHead1 = pHead1.next
            else:
                tmp.next = pHead2
                pHead2 = pHead2.next
            tmp = tmp.next
        if not pHead1:
            tmp.next = pHead2
        if not pHead2:
            tmp.next = pHead1
        return pHead.next

# 递归
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if pHead1 is None:
            return pHead2
        if pHead2 is None:
            return pHead1
        if pHead1.val < pHead2.val:
            pHead1.next = self.Merge(pHead1.next,pHead2)
            return pHead1
        else:
            pHead2.next = self.Merge(pHead1,pHead2.next)
            return pHead2

