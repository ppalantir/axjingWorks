"""
-----------
Qusetion:
https://leetcode.com/problems/valid-parentheses/description
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.

Example 1:

Input: "()"
Output: true
Example 2:

Input: "()[]{}"
Output: true
Example 3:

Input: "(]"
Output: false
Example 4:

Input: "([)]"
Output: false
Example 5:

Input: "{[]}"
Output: true

--------------
Thinking:
1 使用栈遍历输入字符串
2 如果当前字符为左半括号时，则将其压入栈中
3 如果遇到右半括号时，分类讨论：
    1) 如栈不为空且对应左半括号，则取出栈顶元素，继续循环
    2) 若此时栈为空，则直接返回False
    3) 若不为对应的左半括号，返回False

------------
值得注意的是，如果题目要求只有一种括号，那么我们其实可以使用更简洁，更省内存的方式 - 计数器来进行求解，而 不必要使用栈。

事实上，这类问题还可以进一步扩展，我们可以去解析类似HTML等标记语法， 比如
"""

class Solution:
    def isValid(self, s):
        stack = []
        map = {
            "{": "}",
            "[": "]",
            "(": ")"
        }
        for x in s:
            if x in map:
                stack.append(map[x])
            else:
                if len(stack) != 0:
                    top_element = stack.pop()
                    if x != top_element:
                        return False
                    else:
                        continue
                else:
                    return False
        return len(stack) == 0



x = "(){}[]"
S = Solution()
r = S.isValid(x)
print(r)
"""
#-------Conclusion-----------
#将括号映射成字典形式，再使用栈进行判断
#栈（Stack）是一个数据集合，可以理解为只能在一端进行插入或删除的操作列表
#栈的特点:后进先出
栈的基本操作：
* 进栈（压栈）：push
* 出栈：pop
* 取栈顶（看看栈顶是谁）：gettop
"""