'''
@Author: your name
@Date: 2020-02-17 13:14:09
@LastEditTime : 2020-02-17 13:15:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\zijie\众数.py
'''
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        nums.sort()
        return nums[len(nums) // 2]


class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = {}
        for num in nums:                                                # 统计每个数字出现的次数
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        return {v: k for k, v in count.items()}[max(count.values())]    # 字典键值反转，找到出现次数最多的数字

        
s = Solution()
x = [1,2,3, 4]
print(s.majorityElement(x))

