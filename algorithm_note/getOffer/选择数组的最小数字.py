'''
@Author: your name
@Date: 2020-04-07 08:41:16
@LastEditTime: 2020-04-07 08:54:36
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\选择数组的最小数字.py
'''

# 比较
class Solution():
    def minNumberInRotateArray(self, rotateArray):
        length = len(rotateArray)
        if length == 0:
            return 0
        elif length == 1:
            return rotateArray[0]
        else:
            for i in range(length -1):
                if rotateArray[i] > rotateArray[i+1]:
                    return rotateArray[i+1]
            return rotateArray[length-1]

# 二分
class Solution():
    def minNumberInRotateArray(self, rotateArray):
        length = len(rotateArray)
        if length == 0:
            return 0
        elif length == 1:
            return rotateArray[0]
        else:
            left = 0
            right = length - 1
            while left < right:
                mid = (right + left)/2
                if rotateArray[mid] < rotateArray[right]:
                    right = mid
                else:
                    left = mid+1
            return rotateArray[right]
            
            