'''
@Author: your name
@Date: 2020-02-20 17:12:58
@LastEditTime: 2020-02-20 17:13:07
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\识别有效ip地址并统计.py
'''

import re
 
def isLegalIP(IP):
    if not IP or IP == "":
        return False
     
    pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    match = pattern.match(IP)
    if not match:
        return False
     
    nums = IP.split(".")
    for num in nums:
        n = int(num)
        if n<0 or n>255:
            return False
         
    return True
 
def CatagoryIP(IP):
    if not IP or IP == "":
        return False
    nums = IP.split(".")
    # A
    if 126 >= int(nums[0]) >= 1:
        return "A"
    # B
    if 191 >= int(nums[0]) >= 128:
        return "B"
    # C
    if 223 >= int(nums[0]) >= 192:
        return "C"
    # D
    if 239 >= int(nums[0]) >= 224:
        return "D"
    # E
    if 255 >= int(nums[0]) >= 240:
        return "E"
     
    return False
 
def isPrivateIP(IP):
    if not IP or IP == "":
        return False
     
    nums = IP.split(".")
    if int(nums[0]) == 10:
        return True
    if int(nums[0]) == 172:
        if 31 >= int(nums[1]) >= 16:
            return True
    if int(nums[0]) == 192 and int(nums[1]) == 168:
        return True
     
    return False
 
def isLegalMaskCode(Mask):
    if not Mask or Mask == "":
        return False
    if not isLegalIP(Mask):
        return False
         
    binaryMask = "".join(map(lambda x: bin(int(x))[2:].zfill(8), Mask.split(".")))
    indexOfFirstZero = binaryMask.find("0")
    indexOfLastOne = binaryMask.rfind("1")
    if indexOfLastOne > indexOfFirstZero:
        return False
    return True
         
try:
    A, B, C, D, E, Err, P = [0, 0, 0, 0, 0, 0, 0]
    while True:
        s = input()
        IP, Mask = s.split("~")
         
        if not isLegalIP(IP) or not isLegalMaskCode(Mask):
            Err += 1
        else:
            if isPrivateIP(IP):
                P += 1
            cat = CatagoryIP(IP)
            if cat == "A":
                A += 1
            if cat == "B":
                B += 1
            if cat == "C":
                C += 1
            if cat == "D":
                D += 1
            if cat == "E":
                E += 1
except:
    print(A, B, C, D, E, Err, P)
    pass