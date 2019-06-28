# 验证没有基线条件的递归
def recursive_error(i):
    print(i)
    if i < 0:
        print("时间到······")
    else:
        recursive_error(i-1)

recursive_error(100)

# 求一个列表（如:[2, 4, 6]）的和
# 用循环
def for_sum(arr):
    total = 0
    for i in arr:
        total += i

    return total

# Error
def recursive_sum(arr):
    #total = 0
    if arr == []:
        print('-----------')
    else:
        a = arr[0]
        l = arr[1:]
        total = a + recursive_sum(l)
        return total
    

print(for_sum([2, 4, 6]))
print(recursive_sum([2, 4, 6]))

