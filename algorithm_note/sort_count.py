"""
时间复杂度为O（N）来计算频率，O（N + k）以排序顺序输出结果，其中 k 是输入的整数范围，
在本例中为9-1 + 1 = 9。 计数排序（Counting Sort）的时间复杂度为O（N + k），如果 k 很小，那么它就是O（N）。
由于内存限制，当 k 相对较大时，我们将无法执行计数排序（Counting Sort）的计数部分，因为我们需要存储那些 k 个整数出现的次数。

中间数组的作用是用来统计arr中每个元素的出现次数，中间数组的下标就是arr中元素的值，中间数组的值就是arr中该下标值出现的次数
之所以要获取arr的最大值和最小值，是因为需要排序的一定范围的整数不一定是从0开始，此时为避免空间位置浪费，中间数组的长度是是
数列最大值和最小值的差+1，此时最小值作为一个偏移量存在
第一次遍历：用来遍历arr的每个元素，统计每个元素的出现次数，存入中间数组（下标是arr中的元素值，值是该元素在arr中的出现次数）
第二次遍历：遍历中间数组，每个位置的值=当前值+前一个位置的值，用来统计出小于等于当前下标的元素个数
第三次遍历：反向遍历arr的每个元素，找到该元素值在中间数组的对应下标，以这个中间数组的值作为结果数组的下标，将该元素存入结果数组

空间复杂度和时间复杂度
假定原始数列的规模是N，最大值和最小值的差是M,计数排序的时间复杂度是O(N+M)，如果不考虑结果数组，只考虑中间数组大小的话，空间复杂度是O(M)
局限性

当数列的最大和最小值差距过大时，并不适用计数排序
当数列元素不是整数，并不适用计数排序

"""


def count_sort(arr):
    if not isinstance(arr, (list)):
        raise TypeError('error para type')
    
    #获得最大值最小值
    max_num = max(arr)
    min_num = min(arr)

    # 以最大值最小值的差作为中间数值的长度，并构建中间数组，初始化为0
    lenght = max_num - min_num + 1
    tmp_arr = [0 for i in range(lenght)]

    #创建结果list，存放排序完成的结果
    result_arr = list(range(len(arr)))

    # 第一次循环遍历
    for num in arr:
        tmp_arr[num-min_num] += 1

    # 第二次循环遍历
    for i in range(1, lenght):
        tmp_arr[i] = tmp_arr[i] + tmp_arr[i-1]

    # 第三次遍历
    for i in range(len(arr) - 1, -1, -1):
        result_arr[tmp_arr[arr[i] - min_num] -1] = arr[i]
        tmp_arr[arr[i] - min_num] -= 1
    
    return result_arr

if __name__ == "__main__":
    arr = [12,25,26,13,14,25,12,17,18,14]
    print(count_sort(arr))
