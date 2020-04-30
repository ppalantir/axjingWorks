'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-09 18:19:33
@LastEditTime: 2019-10-09 18:21:56
@LastEditors: Please set LastEditors
'''
class Solution:
    def getAllMin(self, X):
        minx = min(X)

        return  minx 



while True:
    try:
        n, m, k = map(int, input().strip().split())
        #print(n, m)
        l_n = []
        l_m = []
        l_n = input().strip().split()
        while len(l_n) == n:
            continue
        l_nn = []
        for i in range(n):
            l_nn.append(l_n[i])

        
        for j in range(m):
            i_m = int(input().strip())
            l_m.append(i_m)
        v_l = Solve(l_nn, l_m, k)
        print(v_l[:k])
    except EOFError:
        break
    except ValueError:
        continue




