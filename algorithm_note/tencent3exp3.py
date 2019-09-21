
def Solve(x_i, y_i):
    X_n = 0
    for i in x_i:
        X_n += i
    return X_n * min(y_i)
    

while True:
    try:
        n, k = map(int, input().strip().split())
        #print(n, m)
        x_i = []
        y_i = []
        for i in range(n):
            xi, yi = map(int, input().strip().split())
            if 1<=xi<=10^6 and 1<=yi<=10^6:
                x_i.append(xi)
                y_i.append(yi)

        if 1<=k<=n<=3*10^5:
            print(Solve(x_i, y_i))

            
            
        

    except EOFError:
        break
    except ValueError:
        continue
