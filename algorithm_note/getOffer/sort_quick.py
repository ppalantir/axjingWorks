
"""
首先，我们分析跑一次分区的成本。
在内部分区（a，i，j）中，只有一个for循环遍历（j-i）次。 由于j可以和 N-1一样大，i 可以低至0，所以分区的时间复杂度是O（N）。
类似于归并排序分析，快速排序的时间复杂度取决于分区（a，i，j）被调用的次数
"""
import random

def quick_sort(arr):
    if len(arr) < 2:
        return arr  #基线条件：为空或者只包括一个元素的数组是“有序”的
    else:
        pivot = arr[0]  # 递归条件
        less = [i for i in arr[1:] if i <= pivot]   # 由所有小于基准值的元素组成的子数组

        greater = [i for i in arr[1:] if i > pivot] # 由所有小于基准值得元素组成的子数组

        return quick_sort(less) + [pivot] + quick_sort(greater)

def random_quick_sort(arr):
    if len(arr) < 2:
        return arr
    else:
        pivot = random.randint(0, len(arr)-1)
        less = [i for i in arr[:pivot] +arr[pivot+1:] if i <= arr[pivot]]

        greater = [i for i in arr[:pivot] + arr[pivot+1:] if i > arr[pivot]]

    return random_quick_sort(less) + [arr[pivot]] + random_quick_sort(greater)


print(quick_sort([2, 3, 5, 1, 8, 1, 9, 4, 7]))

print(random_quick_sort([2, 3, 5, 1, 8, 1, 9, 4, 7]))