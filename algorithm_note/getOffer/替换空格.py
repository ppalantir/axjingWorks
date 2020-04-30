'''
@Author: your name
@Date: 2020-04-02 15:06:07
@LastEditTime: 2020-04-02 15:43:21
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\替换空格.py
'''
class Solution():
    def replace(self, s):

        if s == None or len(s)==0:
            return ""
        else:
            # r = []
            # for i in s:
            #     if i == ' ':
            #         r.append("%20")
            #     else:
            #         r.append(i)
            r = ["20%" if i == " " else i for i in s ]
            return ''.join(r)
s = "adad adads dsfsd sfdf dsfd "
S = Solution()
o = S.replace(s)
print(o)
            