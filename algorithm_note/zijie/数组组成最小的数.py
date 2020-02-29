'''
@Author: your name
@Date: 2020-02-17 13:19:35
@LastEditTime : 2020-02-17 13:24:58
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\zijie\数组组成最小的数.py
'''

'''
def strSort(self, list):
    if len(list) <= 1:
        return list
    left = self.strSort([i for i in list[1:] if (i+list[0]) < (list[0]+i)])
    right = self.strSort([i for i in list[1:] if i+list[0] >= list[0] + i])
    return left+[list[0]]+right
'''


class Solution:
    def PrintMinNumber(self, numbers):
        if numbers is None or len(numbers) == 0:
            return ""
        // 映射为字符串数组
        numbers = map(str, numbers)
        numbers.sort(cmp = lambda x, y : cmp(x + y, y + x))
        return "".join(numbers).lstrip()