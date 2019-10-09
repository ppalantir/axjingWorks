#include <iostream>
using namespace std;
int main(){
	//1创建五个小猪数组
	int arr1[5] = {300, 350, 200, 400, 250};
	//从数组中找到最大值
	int max1=0; //先认定一个最大值为０
	for (int i=0; i<5; i++){
		if (arr1[i] > max1){
			max1 = arr1[i];
			            
			}       
		}
	cout << "小猪的最大值：" << max1 << endl;
    }

