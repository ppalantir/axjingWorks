"""
-----------------
Qusetion:
https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

Example 1:

Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length. Example 2:

Given nums = [0,0,1,1,1,2,2,3,3,4],

Your function should return length = 5, with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.

It doesn't matter what values are set beyond the returned length. Clarification:

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by reference, which means modification to the input array will be known to the caller as well.

Internally you can think of this:

// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}

----------------
Thinking:
使用快速指针来记录遍历的坐标。
1 开始时这两个指针都指向第一个数字
2 如果两个指针指的数字相同，则快指针向前走一步
3 如果不同，则两个指针都向前走一步
4 当快指针走完整个数组后，慢指针当前的坐标加1 就是数组中不同数字的个数

----------------
Key:
1 双指针
这道题如果不要求， O(n)的时间复杂度， O(1)的空间复杂度的话，会更简单。但是这道题所要求的，一般的解题思路是采用双指针

2 如果数据是无序的，就不可以用这种方式了，从这里也可以看出排序算法的基础性和重要性
"""
class Sulation:
    def removeDuplicate(self, nums): # 错误
        """
        :type nums: list[int]
        : rtype: int
        """
        i = 0
        while i < len(nums)-1:
            if nums[i] == nums[i+1]:
                del nums[i]

            else:
                i += 1

            return len(nums)

    def removeDuplicates(self, nums):
        """
        :type nums: List(int)
        :rtype: int
        """
        l = len(nums)
        if l > 0:
            j = 0
            for i in range(1, l):
                if nums[i] == nums[j]:
                    continue
                else:
                    j += 1
                    nums[j] = nums[i]

            return j+1

        else:
            return l




#----------------
S = Sulation()
num = [0, 0, 1, 2, 3, 2, 3, 4, 5]

nums = S.removeDuplicates(num)

print(nums)


"""
Conclusion:
    双指针：一个指针用于迭代（快指针i），另一个指针总是指向下一次添加的位置（慢指针j）
"""
