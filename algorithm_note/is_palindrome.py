"""
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

说明：本题中，我们将空字符串定义为有效的回文串。

示例 1:

输入: "A man, a plan, a canal: Panama"
输出: true
示例 2:

输入: "race a car"
输出: false
"""
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        import re 
        # re.sub(pattern, repl, string, count)用于替换字符串中的匹配项
        s = re.sub('[^a-z0-9]', '', s.lower()) #s.lower()返回将字符串中所有大写字符转换为小写后生成的字符串。
        print(s)
        print(s[::-1])
        # [起始位置：结束位置：步长]                                       
        return s == s[::-1]

S = Solution()
s = "A man, a plan, a canal: Panama"
s1 = "race a car"
a = S.isPalindrome(s)
a1 = S.isPalindrome(s1)
print(a)
print(a1)