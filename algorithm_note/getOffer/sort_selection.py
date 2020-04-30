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

def selection_sort_1(arr):
    
    lenght = len(arr)
    for i in range(lenght-1):
        min_index = i
        for j in range(i + 1, lenght-1):
            if arr[min_index] > arr[j]:
                arr[min_index], arr[j] = arr[j], arr[min_index]
            
        # if i != min_index:
        #     arr[min_index], arr[i] = arr[j], arr[min_index]
    return arr

if __name__ == "__main__":

    print(selection_sort([5, 3, 6, 2, 10]))


    print(selection_sort_1([5, 3, 6, 2, 10]))