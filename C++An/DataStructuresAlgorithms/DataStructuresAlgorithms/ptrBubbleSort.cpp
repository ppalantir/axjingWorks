#include <iostream>
using namespace std;

/**
 *指针冒泡排序函数
 *参数1 数值首地址
 *参数2 数组长度
 **/
void prtBubbleSort(int* arr, int len){
	for (int i= 0; i < len-1; i++){
		for (int j=0; j<len-i-1; j++){
			if (arr[j] > arr[j + 1]){
				int tmp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = tmp;
			}
		}
	}

}

//打印数组
void printArray(int * arr, int len){
	for (int i=0; i < len; i++){
		cout << arr[i] << endl;
	}
}

int main(){
	//创建数组
	int arr[10] = {3, 4, 6, 9, 1,2,10, 8,7,5};
	//数组长度
	int len = sizeof(arr) /sizeof(arr[0]);
	//创建函数
	prtBubbleSort(arr, len);

	// 打印数组
	printArray(arr, len);

//	system("pasue");
	return 0;
}
