def Solve(li):
    box = []
    n = 0
    li_m = []
    li_h = []
    for i in range(len(li)):
        for j in range(len(li)):
            if li[i] > li[j]:
                m = [li[j], li[i]]
            else:
                m = [li[i], li[j]]
            if m not in box and i != j:
                box.append(m)
                if abs(m[0]) > abs(m[1]):
                    m_ = [abs(m[1]), abs(m[0])]
                else:
                    m_ = [abs(m[0]), abs(m[1])]
                if abs(m[0] + m[1]) > abs(m[0] - m[1]):
                    h_ = [abs(m[0] - m[1]), abs(m[0] + m[1])]
                else:
                    h_ = [abs(m[0] + m[1]), abs(m[0] - m[1])]
                if m_[0]>=h_[0] and m_[1]<=h_[1]:
                    n += 1

    return n


while True:
    try:
        num = int(input())
        n_li = map(int, input().strip().split())
        print(Solve(n_li))
        

    except EOFError:
        break
    except ValueError:
        continue