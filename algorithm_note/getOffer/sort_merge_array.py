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
    def merge_0(self, nums1, m, nums2, n):
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

def merge(left_li, right_li):

    result = []
    while len(left_li) > 0 and len(right_li):
        if left_li[0] <= right_li[0]:
            result.append(left_li.pop(0))
        else:
            result.append(right_li.pop(0))
    
    result += left_li
    result += right_li
    return result


def merge_sort(arr):
    
    if len(arr) == 1:
        return arr
    
    mid = len(arr) // 2

    left = arr[:mid]
    right = arr[mid:]

    l1 = merge_sort(left)
    r1 = merge_sort(right)

    return merge(l1, r1)

if __name__ == "__main__":

    S = Solution()
    nums1 = [1,2,3,0,0,0]
    m = 3
    nums2 = [2,5,6]      
    n = 3
    print(S.merge_0(nums1, m, nums2, n))

    """
    从后往前修改比较，并后往前插入

    无论输入的原始顺序如何，归并排序中最重要的部分是其O（N log N）性能保证。
    就这样，没有任何敌手测试用例可以使归并排序对于任何N个元素数组运行比O（N log N）更长的时间。
    因此，归并排序非常适合分类非常大量的输入，因为O（N log N）比前面讨论的O（N2）排序算法增长得慢得多。
    归并排序也是一个稳定的排序算法。 讨论：为什么？
    然而，归并排序有几个不太好的部分。 首先，从零开始实施起来并不容易（但我们不必这样做）。
    其次，它在归并操作期间需要额外的O（N）存储，因此不是真正的存储效率和不到位。顺便说一句，
    如果你有兴趣看看为解决这些（经典）归并排序不那么好的部分做了什么，你可以阅读这个。
    """
    print(merge_sort([7, 2, 6, 3, 8, 4, 5]))