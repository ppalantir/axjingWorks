def chek_str(str_in):
    chek_str='helloworld'
    strl=len(chek_str)
    count=0
    for i in range(len(chek_str)):
        str_tmp=chek_str[i]
        str_in_tmp=str_in[i:len(str_in)]
        if str_tmp in str_in_tmp:
            count=count+1
    if count==strl:
        return True
    else:
        return False

if __name__ == "__main__":
    input_1 = 'heabcllowodefrld'
    input_2 = 'ehlloabcworld'
    print('判断结果：', chek_str(input_1))
    print('判断结果：', chek_str(input_2))