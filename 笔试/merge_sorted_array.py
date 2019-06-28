"""
Qusetion:
https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/261/before-you-start/1109/
给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。

说明:

初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
示例:

输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]

---------------
Thinking:
这道题目其实和基本排序算法中的merge sort非常像，但是 merge sort 很多
合并的时候我们通常是 新建一个数组，这样就很简单。 但是这道题目要求的是原地修改.
    此题的关键点是要求原地修改，可以从后往前修改比较，并后往前插入即可
"""
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead
        """
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
            
        if n > 0:
            nums1[:n] = nums2[:n]

        return nums1


S = Solution()
nums1 = [1,2,3,0,0,0]
m = 3
nums2 = [2,5,6]      
n = 3
print(S.merge(nums1, m, nums2, n))

"""
从后往前修改比较，并后往前插入
"""