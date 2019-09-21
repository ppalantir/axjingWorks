
def Solve(ln_0, ln_1, ln_s):
    if ln_s == "-":
        result = ln_0  - ln_1
    if ln_s == "+":
        result = ln_0  + ln_1
    if ln_s == "*":
        result = ln_0  * ln_1

    return result
    

while True:
    try:
        T = int(input().strip())
        #print(n, m)
        for i in range(T):
            k = int(input().strip())
            li2 = input().strip().split()
            ln_0 = int(li2[0])
            ln_1 = int(li2[1])
            ln_s = li2[2]
        print(Solve(ln_0, ln_1, ln_s))
    except EOFError:
        break
    except ValueError:
        continue
