#include <iostream>
using namespace std;

int main(){
	//实现数组逆置
	//创建数组
	cout << "数组元素逆置前：" << endl;
	int arrA[] = {1, 2, 3, 4, 5};
	for (int i=0; i<5; i++){
		cout <<  arrA[i] << endl;
	}

	//实现逆置
	//１记录起始下标位置
	//２记录下标位置
	//３起始下标位置与结束下标位置互换
	//４起始位置＋＋，结束位置－－
	//循环执行１　起始位置>=结束位置
	
	int startA=0;//起始下标
	int endA = sizeof(arrA) / sizeof(arrA[0]) - 1;//结束下标
	while (startA < endA){
		int tempA = arrA[startA];
		arrA[startA] = arrA[endA];
		arrA[endA] = tempA;
		
		startA++;
		endA--;

	}

	cout << "数组元素逆置后：" << endl;
	for (int i = 0; i < 5; i++){
		cout << arrA[i] <<endl;
	}
}
