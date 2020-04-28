"""
二分查找：
    接受一个有序数组和一个元素。如果指定的元素包含在数组中，这个函数返回其位置
"""

class Solution(object):
    def binary_search(self, list_orderly, target):
        low = 0
        high = len(list_orderly) - 1

        while low <= high:
            mid = (low + high) // 2
            guess = list_orderly[mid]   
            if guess == target:
                return mid
            elif guess > target:
                high = mid - 1
            else:
                low = mid + 1
        return None 

list_orderly = [0, 1, 2, 3, 4, 5]
target = 5
S = Solution()
print(S.binary_search(list_orderly, target))
