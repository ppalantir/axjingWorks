'''
@Author: your name
@Date: 2020-02-17 12:57:31
@LastEditTime : 2020-02-17 13:11:31
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
'''


class Solution(object):
    def myPow(self, x, n):
        if n < 0:
            return self.myPow(1/x, -n)

        if n == 0:
            return 1

        if n==2:
            return x**x

        return self.myPow(self.myPow(x,n/2),2) if not n%2 else x * self.myPow(self.myPow(x,n//2),2)

s = Solution()
print(s.myPow(2, 3))
print(s.myPow(2, -2))
print(s.myPow(2, 2))

