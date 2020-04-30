"""
Qusetion:
https://leetcode.com/problems/maximum-depth-of-binary-tree/description/
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its depth = 3.

--------------------------
Thinking:
树是一种递归的数据结构，因此用递归去解决的时候往往非常容易：
var maxDepth = function(root){
    if (!root) return 0;
    if (root.left && !root.right) return 1;
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));

}

如果使用迭代，首先应该是树的各种遍历，由于求深度，因此使用层次遍历（BFS)非常合适，只记录多少层即可

---------------------
Key:
1 队列
2 队列中用Null（一个特殊元素）来划分每层，或者在对每层迭代之前保存当前队列元素的个数（即当前层所含元素的个数）
3 树的基本操作-遍历-层次遍历（BFS)
"""

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        self.depth = 0
        def dfs(node, level):
            if not node:
                return
            self.depth = max(self.depth, level)
            if node.right:
                dfs(node.right, level+1)
            if node.left:
                dfs(node.left, level+1)
        dfs(root, 1)
        return (self.depth)