
def Solve(use_a, people_f):
    
    for i in range(len(use_a)):
        for j in range(len(people_f)):
            if people_f[j][0] == use_a[i]:
                return use_a[i]
            else:
                return -1

    return result
    

while True:
    try:
        n, m = map(int, input().strip().split())
        #print(n, m)
        use_a = []
        for i in range(n):
            k_i = input().strip()
            use_a.append(k_i)
        people_f = []
        for j in range(m):
            str1, str2 = input().strip().split()
            people_f.append([str1, str2])

            
        print(Solve(use_a, people_f))
    except EOFError:
        break
    except ValueError:
        continue
