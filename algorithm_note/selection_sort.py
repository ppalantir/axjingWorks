"""
将数组元素按从小到大的顺序排序，先编写一个用于找出数组中最小元素的函数，然后进行选择排序
"""


def find_smallest(arr):
    smallest = arr[0]   # 用于存储最小值
    smallest_index = 0  # 用于存储最小元素索引
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index


def selection_sort(arr):
    new_arr = []
    for i in range(len(arr)):
        smallest = find_smallest(arr)
        new_arr.append(arr.pop(smallest))
    return new_arr

print(selection_sort([5, 3, 6, 2, 10]))


