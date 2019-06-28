"""
https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/261/before-you-start/1108/
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。

给定 target = 20，返回 false。

---------
Thinking:
首先要分析一下这个矩阵，每一行递增，行首元素最小；每一列也是递增，列首也是最小。
也就是说假如我判断这一行的行首元素比target大，那么这一行就可以直接排除了，每一列同理。
有了上面的思想，我们就要考虑排除的顺序了，我采取的是按行倒序，按列正序的方式，也就是从左下角入手，向右上角收缩。
当然，也可以选择从右上角入手，向左下角收缩，思路是一样的。

"""

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        rows = len(matrix)
        cols = len(matrix[0]) - 1
        row = 0
        while row < rows and cols >= 0:
            if matrix[row][cols] == target:
                return True
            elif matrix[row][cols] > target:
                cols -= 1
            else:
                row += 1
        return False

matrix = [
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
target = 30
S = Solution()
r = S.searchMatrix(matrix, target)
print(r)