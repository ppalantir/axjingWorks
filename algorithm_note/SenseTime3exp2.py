

def Solve(l_n, l_m, k):
    v_l = []
    for i in range(len(l_n)):
        for j in range(len(l_m)):
            v_ = abs(l_n[i]-l_m[j])
            v_l.append(v_)
    v_l = sorted(v_l)
    return v_l
    


    

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




