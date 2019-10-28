/*
 * @Description: 练习cout
 * @Author: Xjing An
 * @Date: 2019-09-24 09:31:54
 * @LastEditTime: 2019-10-06 19:15:20
 * @LastEditors: Please set LastEditors
 */
#include <iostream> //将文件中的内容条件到程序中，与python的import类似,/ iosteram(input&out stream)中包含了输入输出语句的函数
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iomanip>
#include <windows.h> //window控制终端
#include <string>
#include <limits>
using namespace std;

//exp23 指针和函数
// 实现两个数字交换
void swap23(int a, int b)
{
    int tmp = a;
    a = b;
    b = tmp;
    cout << "swap a23 = " << a23 << endl;
    cout << "swap b23 = " << b23 << endl;
}


int main()
{
    //exp23 指针和函数
    int a23 = 10;
    int b23 = 20;
    swap23(a, b);
    cout << "a23 = " << a23 << endl;
    cout << "b23 = " << b23 << endl;
    
    //exp22 三目运算符,返回的是变量，可以继续赋值
    //语法： 表达式1 ？ 表达式2 ：表达式3；
    //如果表达式1为真，执行表达式2，并返回表达式2的结果
    //如果表达式1为假，执行表达式3，并返回表达式3的结果
    int a22 = 10;
    int b22 = 20;
    int c22 = 0;
    c22 = (a22>b22 ? a22:b22);
    (a22>b22 ? a22 : b22) = 100;
    cout << a22 << endl;
    cout << b22 << endl;
    cout << c22 << endl;
    

    //exp21 bool数据类型
    bool flag21 = true;
    cout << flag21 << "bool类型所占用的内存空间：" << sizeof(flag21) << endl;
    
    //exp20字符串类型
    char str20[] = "I Love You!!!";
    string str201 = "C++：I Love You!!!"; // 包含一个include <string>
    cout << "C风格字符串：" << str20 << endl;
    cout << "C+风格字符串：" << str201 << endl;
    
    //exp19字符型的创建方法、内存空间、常见错误：只能用单引号，只能是单个字符
    char ch19 = 'a';
    cout << "a=" << ch19 << "字符型变量的大小：" << sizeof(ch19) << endl;
    cout << "ASCII值：" << int(ch19) << endl;
    
    //exp18 指针数组用例：
    //单精度vs双精度
    float f1 = 3.14f;
    double f2 = -3.13;
    float f3 = 3e-21;
    cout << "f1=" << f1 << "字符长度：" << sizeof(f1) << endl;
    cout << "f2=" << f2 << "字符长度：" << sizeof(f2) << endl;
    cout << "f3=" << f3 << "字符长度：" << sizeof(f3) << endl;
    double score[] {11, 22, 33, 44, 55};
    double *ptr_score = score;
    
    for (int i = 0; i < 5; i++){
        cout << *ptr_score++ << endl;
    }
    //数组名就是就是数组的首地址
    cout << sizeof(score) << "\t" << sizeof(ptr_score) << endl;
    cout << ptr_score[3] << endl;
    // exp17指针--------------------------------------
    /**
     * 指针是一个值为内存地址的变量（或数据对象）
     * 
     * */
    double num17 = 1024.5;
    //声明一个指针，指向num17变量
    double* ptr_num17 = &num17;
    double &ref_num17 = num17;
    cout << "ptr_num is Value:" << ptr_num17 << "\t" << &num17 << "\t"<< ref_num17 <<  endl;
    cout << "ptr_num17 指向空间的值：" << *ptr_num17 << endl;

    //exp16 数组--------------------------------------
    int num16[] = {1, 2, 4, 'c', 231231};
    cout << num16[2] << endl;
    // 动态录入信息及赋值
    const int N1 = 5;
    int num161[N1];

    for (int i = 0; i < sizeof(num161) / sizeof(int); i++){
        cout << "请输入第" << (i+1) << endl;
        cin >> num161[i];
    }
    for (int i = 0; i<sizeof(num161) / sizeof(int); i++){
        cout << num161[i] << endl;
    }


    //exp15-------------------------------------------
    //break 跳出循环 continue继续下次循环


    //exp14-------------------------------------------
    //for 
    /**
     * for(表达式1；表达式2；表达式3){
          语句；
    }*/
    const int N = 20;
    for (int i=0; i<N; i++){
        cout << "zaibiekangqiao"<< endl;
        if (i == 6){
            break;
        }
    }

    // exp13------------------------------------------
    // while
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
    num9 %= 90; //取模、取余

    cout << num9 << endl;

    //前置递增运算符，后置递增，前置递减，后置递减
    int num_8 = 10;
    int num_8_1 = 10;
    //int num_80 = ++num_8; //前置递增，先让变量+1,然后进行表达式运算
    int num_81 = ++num_8 * 9;
    cout<< "前置递增：" << "num_80" << "前置递增表达式：" << num_81 << endl;
    //int num_82 = num_8++; //后置递增，先进行表达式运算，后让变量+1
    int num_83 = num_8_1++ * 9;
    cout<< "后置递增：" << "num_82" << "后置递增表达式：" << num_83 << endl;

    //int num_84 = --num_8; //前置递增，先让变量+1,然后进行表达式运算
    int num_85 = --num_8 * 9;
    cout<< "前置递减：" << "num_84" << "前置递减表达式：" << num_85 << endl;
    //int num_86 = num_8--; //后置递增，先进行表达式运算，后让变量+1
    int num_87 = num_8-- * 9;
    cout<< "后置递减：" << "num_86" << "后置递减表达式：" << num_87 << endl;
    
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
    //SetConsoleTitle("安相静：WorkSpace");
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
    cout << "type: \t\t" << "************size**************"<< endl;  
    cout << "bool: \t\t" << "所占字节数：" << sizeof(bool);  
    cout << "\t最大值：" << (numeric_limits<bool>::max)();  
    cout << "\t\t最小值：" << (numeric_limits<bool>::min)() << endl;  
    cout << "char: \t\t" << "所占字节数：" << sizeof(char);  
    cout << "\t最大值：" << (numeric_limits<char>::max)();  
    cout << "\t\t最小值：" << (numeric_limits<char>::min)() << endl;  
    cout << "signed char: \t" << "所占字节数：" << sizeof(signed char);  
    cout << "\t最大值：" << (numeric_limits<signed char>::max)();  
    cout << "\t\t最小值：" << (numeric_limits<signed char>::min)() << endl;  
    cout << "unsigned char: \t" << "所占字节数：" << sizeof(unsigned char);  
    cout << "\t最大值：" << (numeric_limits<unsigned char>::max)();  
    cout << "\t\t最小值：" << (numeric_limits<unsigned char>::min)() << endl;  
    cout << "wchar_t: \t" << "所占字节数：" << sizeof(wchar_t);  
    cout << "\t最大值：" << (numeric_limits<wchar_t>::max)();  
    cout << "\t\t最小值：" << (numeric_limits<wchar_t>::min)() << endl;  
    cout << "short: \t\t" << "所占字节数：" << sizeof(short);  
    cout << "\t最大值：" << (numeric_limits<short>::max)();  
    cout << "\t\t最小值：" << (numeric_limits<short>::min)() << endl;  
    cout << "int: \t\t" << "所占字节数：" << sizeof(int);  
    cout << "\t最大值：" << (numeric_limits<int>::max)();  
    cout << "\t最小值：" << (numeric_limits<int>::min)() << endl;  
    cout << "unsigned: \t" << "所占字节数：" << sizeof(unsigned);  
    cout << "\t最大值：" << (numeric_limits<unsigned>::max)();  
    cout << "\t最小值：" << (numeric_limits<unsigned>::min)() << endl;  
    cout << "long: \t\t" << "所占字节数：" << sizeof(long);  
    cout << "\t最大值：" << (numeric_limits<long>::max)();  
    cout << "\t最小值：" << (numeric_limits<long>::min)() << endl;  
    cout << "unsigned long: \t" << "所占字节数：" << sizeof(unsigned long);  
    cout << "\t最大值：" << (numeric_limits<unsigned long>::max)();  
    cout << "\t最小值：" << (numeric_limits<unsigned long>::min)() << endl;  
    cout << "double: \t" << "所占字节数：" << sizeof(double);  
    cout << "\t最大值：" << (numeric_limits<double>::max)();  
    cout << "\t最小值：" << (numeric_limits<double>::min)() << endl;  
    cout << "long double: \t" << "所占字节数：" << sizeof(long double);  
    cout << "\t最大值：" << (numeric_limits<long double>::max)();  
    cout << "\t最小值：" << (numeric_limits<long double>::min)() << endl;  
    cout << "float: \t\t" << "所占字节数：" << sizeof(float);  
    cout << "\t最大值：" << (numeric_limits<float>::max)();  
    cout << "\t最小值：" << (numeric_limits<float>::min)() << endl;  
    cout << "size_t: \t" << "所占字节数：" << sizeof(size_t);  
    cout << "\t最大值：" << (numeric_limits<size_t>::max)();  
    cout << "\t最小值：" << (numeric_limits<size_t>::min)() << endl;  
    cout << "string: \t" << "所占字节数：" << sizeof(string) << endl;  
    // << "\t最大值：" << (numeric_limits<string>::max)() << "\t最小值：" << (numeric_limits<string>::min)() << endl;  
    cout << "type: \t\t" << "************size**************"<< endl;
    

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
    SetConsoleOutputCP(65001);
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