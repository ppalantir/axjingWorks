def exp1():
    while True:
        try:
            a = list(map(int, input().strip().split()))
            if a[0]==0 and a[1]==0:
                break

            print(a[0] + a[1])
        except EOFError:
            break
        except ValueError:
            continue


#---------------------------------------
def exp2():

    t = int(input().strip())
    print(t)
    for i in range(int(t)):
        a, b = map(int, input().strip().split())
        print(a+b)

#exp2()

def exp3():
    while True:
        try:
            a = list(map(int, input().strip().split()))
            if a[0] == 0 and a[1] == 0:
                break
            print(a[0] + a[1])

        except EOFError:
            break
        except ValueError:
            continue

exp3()