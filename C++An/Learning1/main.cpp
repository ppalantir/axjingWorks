/*
 * @Description: 练习cout
 * @Author: Xjing An
 * @Date: 2019-09-24 09:31:54
 * @LastEditTime: 2019-09-26 21:23:13
 * @LastEditors: Please set LastEditors
 */

#include <iostream> //将文件中的内容条件到程序中，与python的import类似,/ iosteram(input&out stream)中包含了输入输出语句的函数
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iomanip>

using namespace std;

int main()
{
    //控制cout显示精度
    //1.强制以小数的方式显示
    cout << fixed;
    // 2.控制显示的精度
    cout << setprecision(2);
    //输出double类型的参数
    double double_num = 10.0 / 3.0;
    cout << double_num <<endl;




    //已知圆柱体的半径和高，求圆柱体的体积
    const float PI = 3.141592653; //const 定义了一个float类型的常量
    float radius = 4.5f;
    float height = 90.0f;
    double volume = PI * pow(radius, 2);
    cout <<volume<<endl;

    cout <<"hello world!!!"<<endl;
    cout <<"最近就业形势不好呀\n";
    cout <<"怎么能找到更好的就业\n";
    cout <<"学习C++\n";
    system("pause");
    
    return 0;
}

//每天语句占一行
//注意分号，大括号
//注意分号，空格
//
/*
 *文件名：
 *描  述：
 *作  者：
 *时  间：
 *版  本：
*/