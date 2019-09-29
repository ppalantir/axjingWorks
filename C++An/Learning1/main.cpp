/*
 * @Description: 练习cout
 * @Author: Xjing An
 * @Date: 2019-09-24 09:31:54
 * @LastEditTime: 2019-09-29 16:56:01
 * @LastEditors: Please set LastEditors
 */
#include <iostream> //将文件中的内容条件到程序中，与python的import类似,/ iosteram(input&out stream)中包含了输入输出语句的函数
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iomanip>
#include <windows.h> //window控制终端
using namespace std;

int main()
{
    // exp13------------------------------------------
    // for
    int i = 0;  //循环变量的初始值
    int sum = 0; //用来保存累加和的变量
    while (i <= 100) //循环变量的条件
    {
        cout << "-----" << endl;
        sum = sum + i; //累加
        i++;    //循环变量更新
        cout << sum << endl;
    }
    
        
    //exp12-------------------------------------------
    int num0 = 5;
    cout << sizeof(num0++) << endl;  //4
    cout << num0 << endl;            //5
    //exp11 ---------------------------------------
    // 菜单选择
    // switch(表达式){case 常量1： 语句；break; } 开关
    int choice = 1;
    switch (choice)
    {
    case 1:
        cout << 1 << endl;
        break;
    case 2:
        cout << 2 << endl;
        break;
    case 3:
        cout << 3 << endl;
        break;
    
    default:
        break;
    }
    // exp10--------------------------------------
    // if  else if  else
    double flower_price; //花的单价
    cout << "黎明前的黑暗逐渐褪去， 海天之间透着一抹亮光，仿佛一把点燃的火把。。。" << endl;
    cin >> flower_price;
    cout << flower_price << endl;
    //exp9----------------------------------------
    //if 控制语句
    double price_louis = 35000.0;
    double price_hemes = 11044.5;
    double price_chanel = 1535.0;
    double total = 0;   // 总价
    double zhekou = 0;  // 折扣
    total = price_chanel + price_hemes + price_louis;
    /**
     *如果购买的三种商品有一种商品单价大于1000
     *或三种商品总额大于5000，折扣率为30%
     *否者没有折扣
     */
    if (price_louis > 1000 || price_hemes >1000 || price_chanel > 1000 || total > 5000){
        cout << "折扣为0.3" << endl;
    }
    else{
        cout << "没有折扣" << endl;
    }

    //exp8----------------------------------------
    //运算符及表达式，“=”赋值运算符
    double salary = 3200.0; //从右往左读
    int num9 = 1024;
    //复合运算符
    num9 += 90;
    num9 -= 90;
    num9 /= 90;
    num9 *= 90;
    num9 %= 90;

    cout << num9 << endl;

    // 关系运算符>, <, >=, <=, ==, !=

    int a = 4, b = 16;
    cout << (a>b) << endl;
    cout << (a<b) << endl;

    // 逻辑运算符 a && b, a并且b; a || b a或b; ! 非
    cout << ("---------------") <<endl;
    cout << (a && b) << endl;
    cout << (a||b) << endl;
    cout << (!a) <<endl;
    cout << ("---------------") <<endl;

    //位运算符，&，|， ~, ^, <<, >>, 按位与、或、非、异或、左移，右移。(其他禁止转换为二进制，运算后再转换为其他进制)
    cout << ("---------------") <<endl;
    cout << (a & b) << endl;
    cout << (a|b) << endl;
    cout << (~a) <<endl;
    cout << (a^b) << endl;
    cout << (4<<a) << endl;
    cout << (a >> 4) << endl;
    cout << ("---------------") <<endl;
    



    //exp7-----------------------------------------
    // 加减乘除和取模
    double attack7 = 272;
    double attack8 = 250;
    double attack9 = 240;

    int num1 = -5, num2 =2;
    num1 = num2++ - --num2; //后置，前置
    cout << num1 << "\t" << num2 << endl;

    cout << num1 / num2 << endl; //除法
    cout << num1 % num2 << endl; //取模


    //cout << setfill("--") << endl;

    cout << setw(8) << attack7 << 
            setw(8) << attack8 <<
            setw(8) << attack9 << endl;
    //exp6-----------------------------------------
    // 输入练习
    int num;
    char ch;
    cout << "请输入一个数据：";
    cin >> ch;
    cout << num << "\t" << ch << endl;


    //exp5-----------------------------------------
    // 修改终端标题：安相静WorkSpace
    SetConsoleTitle("安相静：WorkSpace");
    /**伤害*/
    double value_attack = 57.88;
    /**伤害成长值*/
    double grouth_attack = 4.5;
    /**攻击距离*/
    double distance_attack = 27.58;
    
    cout << "名称：德玛西亚之力"<<endl;
    cout << "伤害：" << value_attack << "(" << distance_attack << ")" <<endl;
    //exp4------------------------------------------
    //sizeof 用来测量数据类型的长度
    cout <<sizeof(double)<<endl;
    cout <<sizeof(long double) << endl;
    cout <<sizeof(3.14f)<< endl;

    //exp3-------------------------------------------
    //控制cout显示精度
    //1.强制以小数的方式显示
    cout << fixed;
    // 2.控制显示的精度
    cout << setprecision(2);
    //输出double类型的参数
    double double_num = 10.0 / 3.0;
    cout << double_num <<endl;

    //exp2-----------------------------------------
    //已知圆柱体的半径和高，求圆柱体的体积
    const float PI = 3.141592653; //const 定义了一个float类型的常量
    float radius = 4.5f;
    float height = 90.0f;
    double volume = PI * pow(radius, 2);
    cout <<volume<<endl;


    //exp1---------------------------------------
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