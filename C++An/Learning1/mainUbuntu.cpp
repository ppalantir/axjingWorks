#include <iostream>

using namespace std;
//exp23 指针和函数
// 实现两个数字交换
void swap26(int *p1, int *p2){
    int tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
    cout << "swap *p1 = " << *p1 << endl;
    cout << "swap *p2 = " << *p2 << endl;
}


/** 
 * exp25 函数的分文件编写:
 * 作用：让代码结构更加清晰
 * 函数份文件编写需要四个步骤：
 * １．创建后缀名为.h的头文件
 * ２．创建后缀名为.cpp的源文件
 * ３．在头文件中写函数的声明
 * ４．在源文件中写函数的定义
 */
 
// exp24定义函数
/**
 * 返回类型　函数名(参数列表){
 * 	函数体语句
 * 	return 表达式
 * }
 */

//函数的声明:提前告诉编译器函数的存在，可以利用函数的声明
//函数的声明
int compare_max(int a, int b);
//比较两个函数，实现两个整型数字进行比较，返回最大值
int compare_max(int a, int b){
	return a>b ? a : b;
}

//两个整形数字相加
int addA(int a, int b){	//定义中的ａ, b成为实际参数，简称实参
	int sumA = a+ b;
	return sumA;
}

// 两个整型数字交换函数
void swapA(int num1, int num2){
	cout << "交换前:" << endl << "num1=" << num1 <<"\t" << "num2=" << num2 << endl;
	int temp;
	temp = num1;
	num1 = num2;
	num2 = temp;
	cout << "交换后：" << endl << "num1:" << num1 << "\t" << "num2=" << num2 <<endl;
}
int main(){

    	//exp26 指针和函数 值传递or地址传递
	//如果是地址传递可以修饰实参
        int a23 = 10;
   	int b23 = 20;
    	swap26(&a23, &b23);
   	cout << "a23 = " << a23 << endl;
  	cout << "b23 = " << b23 << endl;

	//exp24　函数 形式：１无参无反；２无参有反；３有参无反；４有参有反
	int a24 = 9;
	int b24 = 19;

	int c = addA(a24, b24); //调用时a24, b24 为实际参数，简称实参
	cout << "加法函数的和c=" << c << endl;
	
	// 当做值传递时，形参发生改变的时候并不影响实参
	swapA(a24, b24);
	//exp23 二维数组变量名的用途
	int arr23[5][3] ={
		{1, 2, 3},
		{4, 4, 5},
		{3, 5, 8},
		{6, 3, 9},
		{10, 22, 3342}
	};

	//行相加
	for (int i=0; i<5; i++){
		int row_num = 0;
		for (int j=0; j<3; j++){
			row_num += arr23[i][j];
		}
		cout << "行相加" << row_num << endl;
	}

	//列相加
	for (int i=0; i<5; i++){
		int col_num = 0;
		for (int j=0; j<3; j++){
			col_num += arr23[i][j];
		}
		cout << "行相加" << col_num << endl;
	}

	//2可以查看内存首地址
	cout << "二维数组所占内存空间：" << sizeof(arr23) << endl;
	//1可以查看内存空间大小
	cout << "二维数组的第一行所占内存空间:" << sizeof(arr23[0]) << endl;
	cout << "二维数组第一个元素所占内存空间:" << sizeof(arr23[0][0]) << endl;
	cout << "二维数组行数：" << sizeof(arr23) / sizeof(arr23[0]) << endl;
	cout << "二维数组列数：" << sizeof(arr23[0]) / sizeof(arr23[0][0]) << endl;

	cout << "二维数组第一行首地址：" << arr23[0] << endl;
	cout << "二维数组首地址为：" << arr23 << endl;
	cout << "二维数组第一个元素的首地址"<< &arr23[0][0] << endl;

	//exp22 二维数组
	//第一种定义方法
	int arr22[3][4];
	arr22[0][0] = 1;
	arr22[0][1] = 2;
	arr22[0][2] = 3;
	arr22[0][3] = 4;
	arr22[1][0] = 5;
	arr22[1][1] = 6;
	arr22[1][2] = 7;
	arr22[1][3] = 8;
	arr22[2][0] = 9;
	arr22[2][1] = 10;
	arr22[2][2] = 11;
	arr22[2][3] = 12;

	//第二种定义方法
	int arr22_1[5][3] ={
		{1, 2, 3},
		{4, 4, 5},
		{3, 5, 8},
		{6, 3, 9},
		{10, 22, 3342}
	};
	//第三种定义方法
	int arr22_2[5][3] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	//二维数组遍历
	cout << arr22[0][0] <<endl;
	cout << arr22[0][1] <<endl;
	cout << arr22[0][2] <<endl;
	cout << arr22[0][3] <<endl;
	cout << arr22[1][0] <<endl;
	cout << arr22[1][1] <<endl;
	cout << arr22[1][2] <<endl;
	cout << arr22[1][3] <<endl;
	cout << arr22[2][0] <<endl;
	cout << arr22[2][1] <<endl;
	cout << arr22[2][2] <<endl;
	cout << arr22[2][3] <<endl;

	for (int i=0; i<3; i++){
		for (int j=0; j<4; j++){
			cout << arr22[i][j] << endl;
		}
	}

	for (int i = 0; i<5; i++){
		for (int j = 0; j<3; j++){
			cout << arr22_1[i][j] << endl;
			cout << arr22_2[i][j] << endl;
		}
	}
	return 0;
}
