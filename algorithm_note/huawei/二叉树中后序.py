'''
@Author: your name
@Date: 2020-04-01 17:32:06
@LastEditTime: 2020-04-02 16:53:43
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\二叉树中后序.py
'''

'''
题目：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

重建二叉树这里就以 后序遍历 输出展示，这里给出上述博客中描述的几个遍历方法的特点：

特性A，对于前序遍历，第一个肯定是根节点；
特性B，对于后序遍历，最后一个肯定是根节点；
特性C，利用前序或后序遍历，确定根节点，在中序遍历中，根节点的两边就可以分出左子树和右子树；
特性D，对左子树和右子树分别做前面3点的分析和拆分，相当于做递归，即可重建出完整的二叉树
'''

class TreeNode():
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution():
    # 返回构造树的根节点
    def reConstructBinaryTree(self, pre, tin):
        if len(pre) == 0:
            return None
        root = TreeNode(pre[0])
        TinIndex = tin.index(pre[0])
        root.left = self.reConstructBinaryTree(pre[1:TinIndex+1], tin[0:TinIndex])
        root.right = self.reConstructBinaryTree(pre[TinIndex+1:], tin[TinIndex+1:])
        return root

    # 后序遍历
    def PostTraversal(self, root):
        if root != None:
            self.PostTraversal(root.left)
            self.PostTraversal(root.right)
            print(root.val)

pre=[1,2,4,7,3,5,6,8]
tin=[4,7,2,1,5,3,8,6]
S=Solution()
root=S.reConstructBinaryTree(pre,tin)
S.PostTraversal(root)