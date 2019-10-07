/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-10-03 16:08:18
 * @LastEditTime: 2019-10-03 17:29:22
 * @LastEditors: Please set LastEditors
 */
#ifndef DYNAMIC_ARRAY_H
#define DYNAMIC_ARRAY_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//动态增长内存，策略：将存放数据放在堆上
//
typedef struct DYNAMICARRAY{
    /* data */
    int* pAddr; //存放数据地址
    int size; //当前有多少个元素
    int capacity; //容量，当前最大能容纳多少元素
}Dynamic_Array;
//写一系列的相关对于DYNAMICARRAY结构体操作的函数
//初始化
Dynamic_Array* Init_Array();
//插入
void Push_Back_Array(Dynamic_Array* arr, int value);
//根据位置删除
void RemoveByPos_Array(Dynamic_Array* arr, int pos);
//根据值删除
void RemoveByValue_Array(Dynamic_Array* arr, int value);
//查找
int Find_Array(Dynamic_Array* arr, int value);
//打印
void Print_Array(Dynamic_Array* arr);
//释放动态数组的内存
void FreeSpace_Array(Dynamic_Array* arr);
//清空数组
void Clear_Array(Dynamic_Array* arr);
// 获得动态数组容量
int Capacity_Array(Dynamic_Array* arr);
// 获得动态数组当前元素个数
int Size_Array(Dynamic_Array* arr);
//根据位置获取某个元素
int At_Array(Dynamic_Array* arr, int pos);
#endif