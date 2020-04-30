'''
@Author: Anxjing.AI
@Date: 2020-04-30 18:19:47
@LastEditTime: 2020-04-30 19:55:57
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\二叉树的镜像.py
'''
'''
题目描述
操作给定的二叉树，将其变换为源二叉树的镜像。
输入描述:
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5

思路：
补充：
数学归纳法：
证明当n等于任意一个自然数时某命题成立。证明分下面两步：
1. 证明当n= 1时命题成立。
2. 假设n=m时命题成立---->那么可以推导出在n=m+1时命题也成立。（m代表任意自然数）

‘先解决最小问题，然后一步一步扩大，设置递归条件’
    1. 假设左右子树自身已镜像；
    2. 交换左右子树：root.left, root.right;
    3. 进入递归处理子问题
'''

class Solution:
    def Mirror(self, root):
        # write code here
        if not root:
            return root
        node=root.left
        root.left=root.right
        root.right=node
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root