"""
给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
示例 3：

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false

--------------
Thinking:
使用动态规划求解，首先选择存储什么历史信息以及用什么数据结构来存储信息。然后是最重要的递推式，
就是如从存储的历史信息中得到当前步的结果。最后起始条件的值。
设置一个动态数组dp[]这个数组保存的是之前的单词是否能够组成单词，之前没有使用动态规划，导致s中出现重叠的情况，
就不能很好地分词，这个代码的思路是对s进行遍历，判断条件是如果s[:i+1]也就是s的前i+1个字符在dict中出现，并且dp[i] == 1
说明这个单词有效
"""
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp = [True]
        lenght = len(s)
        i = 1
        while i < lenght + 1:
            dp.append(False)
            j = i - 1
            while j >= 0:
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
                j -= 1
            i += 1

        return dp[len(s)]

S = Solution()
s = "leetcode"
wordDict = ["leet", "code"]
# s = "applepenapple", 
# wordDict = ["apple", "pen"]
# s = "catsandog"
# wordDict = ["cats", "dog", "sand", "and", "cat"]
print(S.wordBreak(s, wordDict))
