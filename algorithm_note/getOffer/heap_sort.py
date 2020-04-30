# 交换堆顶元素和最后元素
def swap_param(L, i, j):
    L[i], L[j] = L[j], L[i]
    return L
# 堆调整函数
def heap_adjust(L, start, end):
    temp = L[start]

    i = start
    j = 2 * i

    while j <= end:
        if (j < end) and (L[j] < L[j + 1]):
            j += 1
        if temp < L[j]:
            L[i] = L[j]
            i = j
            j = 2 * i
        else:
            break
    L[i] = temp
def heap_sort(L):
    L_length = len(L) - 1

    first_sort_count = int(L_length / 2)
    for i in range(first_sort_count):
        heap_adjust(L, first_sort_count - i, L_length)

    for i in range(L_length - 1):
        L = swap_param(L, 1, L_length - i)
        heap_adjust(L, 1, L_length - i - 1)

    return [L[i] for i in range(1, len(L))]

l=[1,2,6,8,1,4,112,54,8865,24]
print(heap_sort(l))