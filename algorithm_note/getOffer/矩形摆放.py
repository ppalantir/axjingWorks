'''
@Author: your name
@Date: 2020-04-08 14:44:50
@LastEditTime: 2020-04-08 15:08:28
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\矩形摆放.py
'''
class solution():

    def rectCover(self, number):
        if number <=1:
            return number
        if number*2 == 2:
            return 1
        elif number * 2 == 4:
            return 2
        else:
            return self.rectCover(number-1)+self.rectCover(number-2)
class solutions():

    def rectCover(self, number):
        res= [0,1,2]
        while len(res)<=number:
            res.append(res[-1]+res[-2])
        return res[-1]

S = solution()
print(S.rectCover(900))

SS=solutions()
print(SS.rectCover(900))