<!--
 * @Author: your name
 * @Date: 2020-04-30 21:45:59
 * @LastEditTime: 2020-04-30 21:47:08
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \axjingWorks\JingNotebook\dataStructAlgor\剑指offer顺时针打印矩阵.md
 -->

题目描述
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

思路：
1. 模拟模仿逆时针旋转，做取第一行操作；
2. 输出并删除第一行
3. 逆时针旋转剩下矩阵
4. 重复上述操作

```python
class Solution():
    def printMatrix(self, matrix):
        res = []
        while matrix:
            res += matrix.pop(0)
            if not matrix or not matrix[0]:
                break
            matrix = self.turn(matrix)
        return res
    
    def turn(self, matrix):
        num_r = len(matrix)
        num_c = len(matrix[0])
        newmat = []
        for i in range(num_c):
            newmat2 = []
            for j in range(num_r):
                newmat2.append(matrix[j][i])
            newmat.append(newmat2)
        newmat.reverse()
        return newmat
```