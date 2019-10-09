#include <iostream>

using namespace std;

int main(){
	//利用冒泡排序实现升序排序
	int arrA[9] = {4, 2, 8, 0, 7, 5, 1, 3, 9};

	cout << "排序前：" << endl;
	for (int i=0; i<9; i++){
		cout << arrA[i] <<endl;
	}

	//冒泡排序
	//总共的轮数等于元素个数－１
	for (int i = 0; i < 9-1; i++){
		//每次对比次数＝元素个数－当前轮数－１
		for (int j = 0; j<9-i-1; j++){
			//如果第一个数字比第二个大，交换两个数字
			if (arrA[j] > arrA[j+1]){
				int temp = arrA[j];
				arrA[j] = arrA[j+1];
				arrA[j+1] = temp;
			}
		}

	}

	cout << "排序后的结果:" << endl;
	for (int i=0; i<9; i++){
		cout << arrA[i] <<endl;
	}
}
