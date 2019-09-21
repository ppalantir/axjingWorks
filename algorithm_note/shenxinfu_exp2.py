

def Solve(input1, input2):
    
    boxs = 0
    for i in input1:
        boxs = boxs + int(i)
        #print(boxs)
    car = boxs / input2
    n_ = boxs // input2
    if car-n_ == 0:
        cars = n_
    else:
        cars = n_ + 1
    #print(cars)
    return cars





if __name__ == "__main__":

    input1 = input("请输入每个人运送箱子数:").split()
    input2 = int(input("请输入运力："))


    # x1 = []
    # for i in input1:
    #     x1.append(int(i))
    #print(x1)
    #print(int(input2),"=====")
    car = Solve(input1, input2)
    print(car)
