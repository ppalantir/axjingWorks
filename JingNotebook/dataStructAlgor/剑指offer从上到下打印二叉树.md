<!--
 * @Author: your name
 * @Date: 2020-05-01 08:37:06
 * @LastEditTime: 2020-05-01 08:38:12
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \axjingWorks\JingNotebook\dataStructAlgor\剑指offer从上到下打印二叉树.md
 -->

- 题目描述
从上往下打印出二叉树的每个节点，同层节点从左至右打印。

    - 思路：
        1. 使用两个数组，queue存放节点，queue相当于队列， 把每一层的节点都放进去，
        2. 开始 就是循环节点的左右子节点，然后放在队列后面，
        3. result存放节点的值用于最后返回打印
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def PrintFromTopToBottom(self, root):
        if  not root:
            return []
        queue = [root]
        result=[]
        while len(queue)>0:
            node = queue.pop(0)
            result.append(node.val)
            if node.left != None:
                queue.append(node.left)
            if node.right!=None:
                queue.append(node.right)
        return result
```