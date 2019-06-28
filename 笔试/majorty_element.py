"""
https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/261/before-you-start/1107/
给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在众数。

示例 1:

输入: [3,2,3]
输出: 3
示例 2:

输入: [2,2,1,1,1,2,2]
输出: 2

--------------
Thinking:
    摩尔投票法：充分利用了在数组中出现次数大于n/2的元素这一条件，时间复杂度O(n)
    从第一个数开始，先令count=1，遇到相同的就加1，遇到不同的就减1，减到0就重新换下个数开始计数，到最后是哪个数，那么那个数就是众数。
"""
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        major = nums[0]
        count = 1
        n = len(nums)
        del nums[0]
        for i in nums:
            if count == 0:
                count += 1
                major = i
            elif major == i:
                count += 1
            else:
                count -= 1
            return major
        