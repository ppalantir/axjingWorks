<!--
 * @Author: Anxjing.AI
 * @Date: 2020-04-30 17:58:19
 * @LastEditTime: 2020-04-30 18:03:39
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \axjingWorks\JingNotebook\剑指offer_树的子结构.md
 -->

[牛客][https://www.nowcoder.com/practice/6e196c44c7004d15b1610b9afca8bd88?tpId=13&tqId=11170&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking']

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

思路：

1. 在A查找与B根节点一样的节点，ps:查找书的节点采用递归
2. 判断树A中以R为根节点的子树是不是和树B有相同的结构

    2.1 递归，如果节点R的值与树B的根节点不同，则以R为根节点的子树和树B不具有相同节点

    2.2 如果节点R的值与树B的根节点相同，则递归判断各自的左右节点值是否相同
3. 递归的终止条件：达到了A或B的叶节点

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        result = False
        if pRoot1 != None and pRoot2 != None:
            if pRoot1.val == pRoot2.val:
                result = self.DoesTree1haveTree2(pRoot1, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.left, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.right, pRoot2)
        return result
    # 用于递归判断树的每个节点是否相同
    # 需要注意的地方是: 前两个if语句不可以颠倒顺序
    # 如果颠倒顺序, 会先判断pRoot1是否为None, 其实这个时候pRoot2的结点已经遍历完成确定相等了, 但是返回了False, 判断错误
    def DoesTree1haveTree2(self, pRoot1, pRoot2):
        if pRoot2 == None:
            return True
        if pRoot1 == None:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.DoesTree1haveTree2(pRoot1.left, pRoot2.left) and self.DoesTree1haveTree2(pRoot1.right, pRoot2.right)
```