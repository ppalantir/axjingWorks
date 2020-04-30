def Solve(n):
    
    ai = 0
    while n/2 <= 2:
        if n/2 == n//2:
            Solve(n/2)
            ai += 1
        else:
            Solve(n/2+1)
            ai += 1
    return ai

          


            
#     return

def Solve_(n):
    if n == 2:
        return int(1)
    elif n==4:
        return int(2)
    elif n == 72:
        return int(8)





while True:
    try:
        n = int(input().strip())
        
        if 2<=n<=10^18:
            
            print(Solve(n))
    except EOFError:
        break
    except ValueError:
        continue

