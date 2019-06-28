def quick_sort(arr):
    if len(arr) < 2:
        return arr  #基线条件：为空或者只包括一个元素的数组是“有序”的
    else:
        pivot = arr[0]  # 递归条件
        less = [i for i in arr[1:] if i <= pivot]   # 由所有小于基准值的元素组成的子数组

        greater = [i for i in arr[1:] if i > pivot] # 由所有小于基准值得元素组成的子数组

        return quick_sort(less) + [pivot] + quick_sort(greater)


print(quick_sort([2, 3, 5, 1, 8, 1, 9, 4, 7]))