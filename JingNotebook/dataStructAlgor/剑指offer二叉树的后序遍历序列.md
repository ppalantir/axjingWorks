<!--
 * @Author: your name
 * @Date: 2020-05-04 20:16:40
 * @LastEditTime: 2020-05-04 20:18:06
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \axjingWorks\JingNotebook\dataStructAlgor\剑指offer二叉树的后序遍历序列.md
 -->

- 题目描述

输入一个非空整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

- 思路：

后序遍历 的序列中，最后一个数字是树的根节点，
数组中前面的数字可以分为两部分：第一部分是左子树节点 的值，都比根节点的值小；
第二部分 是右子树 节点的值，都比 根 节点 的值大，后面用递归分别判断前后两部分 是否 符合以上原则

BST的后序序列的合法序列是，对于一个序列S，最后一个元素是x （也就是根），如果去掉最后一个元素的序列为T，
那么T满足：T可以分成两段，前一段（左子树）小于x，后一段（右子树）大于x，且这两段（子树）都是合法的后序序列。完美的递归定义 : ) 。
'''
```python
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if sequence==None or len(sequence)==0:
            return False
        length=len(sequence)
        root=sequence[length-1]
        # 在二叉搜索 树中 左子树节点小于根节点
        for i in range(length):
            if sequence[i]>root:
                break
        # 二叉搜索树中右子树的节点都大于根节点
        for j  in range(i,length):
            if sequence[j]<root:
                return False
        # 判断左子树是否为二叉树
        left=True
        if  i>0:
            left=self.VerifySquenceOfBST(sequence[0:i])
        # 判断 右子树是否为二叉树
        right=True
        if i<length-1:
            right=self.VerifySquenceOfBST(sequence[i:-1])
        return left and right
```
**【欢迎关注、点赞、收藏、私信、交流】共同学习进步**
