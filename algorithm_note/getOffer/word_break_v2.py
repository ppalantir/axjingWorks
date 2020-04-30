"""
给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符
串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有
这些可能的句子。

说明：

分隔时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
示例 1：

输入:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
输出:
[
  "cats and dog",
  "cat sand dog"
]
示例 2：

输入:
s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
输出:
[
  "pine apple pen apple",
  "pineapple pen apple",
  "pine applepen apple"
]
解释: 注意你可以重复使用字典中的单词。
示例 3：

输入:
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
输出:
[]

-----------------
Thinking
任然采用动态规划的方法，只不过在每一次动态规划过程中不仅记录是否可以拆分，还要把所有的拆分形式记录下来，
但在最后提交时，卡在了第31个测试集上，原因是内存消耗过多。无奈只能用其他的方法。再阅读了有关资料后，
使用dp(动态规划)判断能否分割，如果可以分割采用DFS分割。DFS分割的思路即每次从字典中找可以与字符串开头匹配的，
并将后续的传入做下一次迭代。
简单分析了一下性能后，的确后者方法无论在空间或者时间上都有巨大优势，动态规划适用于拆分的特例（如能否拆分的判断），
对于遍历的问题还需要用搜索到方法。
深度优先搜索（缩写DFS）有点类似广度优先搜索，也是对一个连通图进行遍历的算法。它的思想是从一个顶点V0开始，沿着一条路一直走到底，
如果发现不能到达目标解，那就返回到上一个节点，然后从另一条路开始走到底，这种尽量往深处走的概念即是深度优先的概念。

"""
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        tmp = wordDict[:]
        wordDict = {}
        for w in tmp:
            wordDict[w] = 1
        res = []
        tmp = []
        def check(s): #判断是不是可以拆分
            l = len(s)
            dp = [False]*(l + 1)
            dp[0] = True
            for i in range(l):
                for j in range(i, -1, -1):
                    if dp[j] and s[j:i+1] in wordDict:
                        dp[i+1] = True
                        break
            return dp[l]

        def DFS(s):
            if not s: # s为空拆分到头
                res.append(tmp[:])
                return
            for word in wordDict:
                l = len(word)
                if word == s[:l]:
                    tmp.append(word)
                    DFS(s[l:])
                    tmp.pop()
        if check(s):
            DFS(s)
        r = []
        for e in res:
            r.append(" ".join(e)) # 转化形式
        return r


S = Solution()
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
print(S.wordBreak(s, wordDict))
