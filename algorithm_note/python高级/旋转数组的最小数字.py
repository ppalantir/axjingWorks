'''
@Author: your name
@Date: 2020-02-20 09:49:10
@LastEditTime: 2020-02-20 09:49:45
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\python高级\旋转数组的最小数字.py
'''


'''
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
'''

# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # 可以使用min()函数， 但是在不适用min函数的情况下，思路应该是二分查找
        # 下面使用 递归实现2分查找，其中递归的时候必须使用return！！！ 不然会返回none
        # write code here 
        start = 0
        end = len(rotateArray) - 1
        mid = int((start + end) / 2)
        if len(rotateArray) == 2:
            if rotateArray[0]>rotateArray[1]:
                return rotateArray[1]
            else:
                return rotateArray[0]

        elif rotateArray[mid] > rotateArray[end]:
            return self.minNumberInRotateArray(rotateArray[mid:end + 1])
        elif rotateArray[mid] < rotateArray[start]:
            return self.minNumberInRotateArray(rotateArray[start:mid + 1])