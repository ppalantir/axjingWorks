"""
https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/261/before-you-start/1106/
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4

----------------------
Thinking:
   与、或、异或运算 
   0异或任何数都不变，任何数与自己异或都为0， 异或满足加法交换律和结合律
   a ^ b ^ a = b
"""
class Solution:
    def single_number(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for i in nums:
            res ^= i
            print(res)

        return res

S = Solution()
nums = [4,1,2,1,2]
res = S.single_number(nums)
print(res)