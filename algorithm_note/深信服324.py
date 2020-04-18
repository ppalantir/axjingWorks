'''
@Author: your name
@Date: 2020-03-24 15:40:00
@LastEditTime: 2020-03-24 17:59:56
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\合并排序链表.py
'''
class Solution(object):
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        queue = [root]
        res = []
        flag = 1
        while queue:
            templist = []
            length = len(queue)
            for i in range(length):
                temp = queue.pop(0)
                templist.append(temp.val)
                if temp.left:
                    queue.append(temp.left)
                if temp.right:
                    queue.append(temp.right)
            if flag == -1:
                templist = templist[::-1]
            res.append(templist)
            flag *=-1
        return res


class Solution(object):
    def zvisit(self, root):
        queue = [root]
        res = []
        flag = 1
        if root == None:
            return res
        
        while queue:
            templist = []
            length = len(queue)
            for i in range(length):
                node = queue.pop(0)
                templist.append(node.val)
                if temp.left:
                    queue.append(temp.left)
                if temp.right:
                    queue.append(temp.right)
            res.append(templist)
        temp = []
        for i in range(len(res)):
            if i%2 == 0:
                temp.append(res[i])
            else:
                temp.append(res[i][::-1])
        return res
        
            

def Solution(n, m):
    if n == 0:
        return None
    if n == 1:
        return 1

    if n>1:
        ll = [i+1 for i in range(n)]
        point = 0
        while len(ll)>1:
            point -= 1
            for j in range(m):
                point +=1
                if point == len(ll):
                    point = 0
            ll.remove(ll[point])
        out = ll[0]
    return out

while True:
    try:
        nm = input.split(" ")
        n = int(nm[0])
        m = int(nm[1])
        out = Solution(n,m)
        print(out)
    except:
        break
        pass

from collections import deque

class Solution(object):
    def zvisit(self, root):
        ret = []
        level_list = deque()
        if root is None:
            return []
        node_queue = deque([root, None])
        is_order_left = True
        while len(node_queue)>0:
            curr_node = node_queue.popleft()
            if curr_node:
                if is_order_left:
                    level_list.append(curr_node.val)
                else:
                    level_list.appendleft(curr_node.val)
                
                if curr_node.left:
                    node_queue.append(curr_node.left)
                if curr_node.right:
                    node_queue.append(curr_node.right)
            else:
                if len(node_queue)>0:
                    node_queue.append(None)
                level_list = deque()
                is_order_left = not is_order_left
        return ret
        
        while queue:
            templist = []
            length = len(queue)
            for i in range(length):
                node = queue.pop(0)
                templist.append(node.val)
                if temp.left:
                    queue.append(temp.left)
                if temp.right:
                    queue.append(temp.right)
            res.append(templist)
        temp = []
        for i in range(len(res)):
            if i%2 == 0:
                temp.append(res[i])
            else:
                temp.append(res[i][::-1])
        return res

def Solu(n,x,lis):
    res_lis = []
    for i in range(len(lis)):
        allt=lis[i][0]+lis[i][1]+sum[res_lis]-x
        if lis[i][0]+lis[i][1]+sum[res_lis]>x:
            res_lis.append(lis[i][0]+lis[i][1]-x)
        else:
            res_lis=[]
    return sum[res_lis]

while True:
    try:
        nx = input().split(" ")
        n, x = int(nx[0]), int(nx[1])
        lis = []
        for i in range(n):
            listmp = []
            tmp = input().split(" ")
            for j in range(len(tmp)):
                listmp.append(int(tmp[j]))
            lis.append(listmp)
        res = Solu(n,x,lis)
        print(res)
    except:
        break