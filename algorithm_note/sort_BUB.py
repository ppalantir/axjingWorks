"""
最优时间复杂度：O(n) （表示遍历一次发现没有任何可以交换的元素，排序结束。）
最坏时间复杂度：O(n^2)
稳定性：稳定
"""
def BUB(num_list):
    lenght = len(num_list)
    
    for i in range(lenght - 1):
        count = 0
        for j in range(0, lenght - i -1):
            if num_list[j] > num_list[j+1]:
                num_list[j], num_list[j+1] = num_list[j+1], num_list[j]
                count += 1
        if count == 0:
            break


    return num_list

if __name__ == "__main__":
    num_list = [29,10,14,37,14,200, 3]
    print(BUB(num_list))
