'''
@Author: your name
@Date: 2020-02-20 08:32:26
@LastEditTime: 2020-02-20 08:32:46
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\遍历链表.py
'''
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        out=[]
        head=listNode
        while head!=None:
            out.append(head.val)
            head=head.next
        out.reverse()
        return out
