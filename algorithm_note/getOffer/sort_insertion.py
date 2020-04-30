"""
插入排序（英语：Insertion Sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，
对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，在从后向
前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。

外循环执行N-1次，这很明显。
但内循环执行的次数取决于输入：
在最好的情况下，数组已经排序并且（a [j]> X）总是为假所以不需要移位数据，并且内部循环运行在O（1），
在最坏的情况下，数组被反向排序并且（a [j]> X）始终为真插入始终发生在数组的前端，并且内部循环以O（N）运行。

最优时间复杂度：O(n) （升序排列，序列已经处于升序状态） 
最坏时间复杂度：O(n2) 
稳定性：稳定
"""

def insertion(arr):
    lenght = len(arr)
    for i in range(1, lenght):
        for j in range(i, 0, -1):
            if arr[j] < arr[j - 1]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
            else:
                break

    return arr
    
if __name__ == "__main__":
    arr = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(arr)
    arr = insertion(arr)
    print(arr)
