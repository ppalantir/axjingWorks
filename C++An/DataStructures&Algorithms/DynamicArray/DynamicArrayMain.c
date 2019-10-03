/*
 * @Description: In User 
 * @Author: XJingAn
 * @Date: 2019-10-03 16:10:44
 * @LastEditTime: 2019-10-03 17:30:16
 * @LastEditors: Please set LastEditors
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DynamicArray.h"

void test01(){
    //初始化数组
    Dynamic_Array* myArray = Init_Array();
    //插入数组
    for (int i = 0; i < 10; i++){
        Push_Back_Array(myArray, i);
    }
    //打印
    Print_Array(myArray);
    //销毁
    FreeSpace_Array(myArray);
}

int main(void){
    test01;
    system("pause");
    return 0;
}

