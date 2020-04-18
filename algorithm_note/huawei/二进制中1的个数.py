'''
@Author: your name
@Date: 2020-04-08 15:23:24
@LastEditTime: 2020-04-08 15:32:34
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\二进制中1的个数.py
'''
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        if n < 0:
            n = n & 0xffffffff
            
        while n:
            count += 1
            n = (n - 1) & n
        return count

class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        if n < 0:
            n = n & 0xffffffff
        while n:
            count += 1
            n = (n - 1) & n
        return count

S=Solution()
print(S.Numberof1(111000))