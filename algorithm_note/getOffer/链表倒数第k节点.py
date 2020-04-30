'''
@Author: your name
@Date: 2020-03-24 15:28:27
@LastEditTime: 2020-03-24 15:35:28
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
'''

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if head==None or k<=0:
            return None
        #设置两个指针，p2指针先走（k-1）步，然后再一起走，当p2为最后一个时，p1就为倒数第k个数
        p2=head
        p1=head
        #p2先走，走k-1步，如果k大于链表长度则返回 空，否则的话继续走
        while k>1:
            if p2.next!=None:
                p2=p2.next
                k-=1
            else:
                return None
        #两个指针一起 走，一直到p2为最后一个,p1即为所求
        while p2.next!=None:
            p1=p1.next
            p2=p2.next
        return p1

if __name__ == "__main__":
    S = Solution()
    o = S.FindKthToTail(3,1)
    print(o)