'''
在一个二维数组中（每个一维数组的长度相同）,每一行都按照从左到右递增
的顺序排序，每一列都按照从上到下递增的顺序排
序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数
'''
class Solution:
    # array二维列表
    def find(self, target, array):
        xend = len(array) - 1
        yend = len(array[0]) - 1

        x = 0
        while x <= xend and yend >= 0:
            if array[x][yend] == target:
                return True

            elif array[x][yend] > target:
                yend -= 1
            else:
                x += 1

        return False

if __name__ == "__main__":
    array = [[1,2,5,7,9],
             [2,4,6,8,10],
             [3,5,7,9,11],
             [4,6,8,10,12]]
    S = Solution()        
    print(S.find(22, array))