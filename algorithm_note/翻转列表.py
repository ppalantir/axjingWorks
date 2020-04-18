'''
@Author: your name
@Date: 2020-03-24 15:38:23
@LastEditTime: 2020-03-24 15:38:24
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\翻转列表.py
'''
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if pHead==None or pHead.next==None:
            return pHead
        pre = None
        cur = pHead
        while cur!=None:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre