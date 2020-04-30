"""
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。

示例:

输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]

-------------
Thinking:
若是按照之前131题回文分割的方法，求出所有的可能，再统计出最小的分割数，会超出时间限制无法通过。此题不要求求出具体分割方案，
可以考虑使用动态规划求解。
假设输入字符串为’aabac’:
|a 　|a　 |b　 |a　 |c　 |
0　１　 ２　 ３　 4　 5
我们可以创建一个数组res来存储每个位置的初始最小分割次数，也就是当字符串中完全没有回文串时的最大分割次数，即[-1,0,1,2,3,4]。
如何进一步更新每个位置需要的分割次数呢？我们先来依次遍历每个分割位，实验观察一下res的变化。
在位置0时，不做更新。
在位置1时，不做更新。
在位置2时，发现’aa’可以构成回文串，res更新为[-1,0,0,2,3,4]。
在位置3时，最小分割为’aa’和’b’，需要一次分割，res更新为[-1,0,0,1,3,4]。
在位置4时，最小分割为’a’和’aba’，需要一次分割，res更新为[-1,0,0,1,1,4]。
在位置5时，最小分割为’a’，‘aba’和’c’，需要两次分割，res更新为[-1,0,0,1,1,2]。
可以得出一下结论，当我们在分割位i时，若s[j:i]为回文串，那么res[i]的分割次数为在分割位j的次数res[j] + 1。题目要求是最少的次数，
因此j遍历0到i-1的所有位置,求出res[i]的最小值。最后我们只要返回res[-1]就是我们要的答案。
到这里此题我们差不多已经解决一半了，为什么是一半呢？因为如果每次都判断s[j:i]是否是回文串会有很多次重复判断，
从而导致超时。解决方法是空间换时间，创建一个二维数组if_palindrome来存储所有子字符串是否是回文串。
这个二维数组的赋值操作可以在上述循环中同时进行。若s[j:i]首尾相等，中间部分也是回文串或者s[j:i]的长度小于等于2(即中间部分为空)，
则s[j:i]为回文串，if_palindrome[j][i] = True。

"""
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        def is_palindrome(substring): #判断是否是回文字符串
            begin, end = 0, len(substring)-1
            while begin < end:
                if substring[begin] != substring[end]:
                    return False
                begin += 1
                end -= 1
            return True

        def search(totaly_string, lists, index):
            if index == len(totaly_string):
                self.result.append(lists)

            for i in range(index, len(totaly_string)):
                substring = totaly_string[index:i+1]
                if not is_palindrome(substring):
                    continue
                search(totaly_string, lists + [substring], i+1)

        self.result = []
        search(s, [], 0)
        return self.result

S = Solution()
print(S.partition("aab"))
print(S.partition("amanaplanacanalpanama"))
