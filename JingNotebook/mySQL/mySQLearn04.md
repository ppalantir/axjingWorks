# mySQL入门04 内置函数
## 字符串函数
1. 查看字符的ascii码值ascii(str)，str是空串时返回0
```
select ascii('a');
```
2. 查看ascii码值对应的字符char(数字)
```
select char(97);
```
3. 拼接字符串concat(str1,str2...)
```
select concat(12,34,'ab');
```
4. 包含字符个数length(str)
```
select length('abc');
```
5. 截取字符串
```
left(str,len)返回字符串str的左端len个字符
right(str,len)返回字符串str的右端len个字符
substring(str,pos,len)返回字符串str的位置pos起len个字符
select substring('abc123',2,3);
```
6. 去除空格
```
ltrim(str)返回删除了左空格的字符串str
rtrim(str)返回删除了右空格的字符串str
trim([方向 remstr from str)返回从某侧删除remstr后的字符串str，方向词包括both、leading、trailing，表示两侧、左、右
select trim('  bar   ');
select trim(leading 'x' FROM 'xxxbarxxx');
select trim(both 'x' FROM 'xxxbarxxx');
select trim(trailing 'x' FROM 'xxxbarxxx');
```
7. 返回由n个空格字符组成的一个字符串space(n)
```
select space(10);
```
8. 替换字符串replace(str,from_str,to_str)
```
select replace('abc123','123','def');
```
9. 大小写转换，函数如下
```
lower(str)
upper(str)
select lower('aBcD');
```
## 数学函数
1. 求绝对值abs(n)
```
select abs(-32);
```
2. 求m除以n的余数mod(m,n)，同运算符%
```
select mod(10,3);
select 10%3;
```
3. 地板floor(n)，表示不大于n的最大整数
```
select floor(2.3);
```
4. 天花板ceiling(n)，表示不小于n的最大整数
```
select ceiling(2.3);
```
5. 求四舍五入值round(n,d)，n表示原数，d表示小数位置，默认为0
```
select round(1.6);
```
6. 求x的y次幂pow(x,y)
```
select pow(2,3);
```
7. 获取圆周率PI()
```
select PI();
```
8. 随机数rand()，值为0-1.0的浮点数
```
select rand();
```
9. 日期时间函数
获取子值，语法如下
```
year(date)返回date的年份(范围在1000到9999)
month(date)返回date中的月份数值
day(date)返回date中的日期数值
hour(time)返回time的小时数(范围是0到23)
minute(time)返回time的分钟数(范围是0到59)
second(time)返回time的秒数(范围是0到59)
select year('2016-12-21');
```
10. 日期计算，使用+-运算符，数字后面的关键字为year、month、day、hour、minute、second
```
select '2016-12-21'+interval 1 day;
```
日期格式化date_format(date,format)，format参数可用的值如下

获取年%Y，返回4位的整数
*　获取年%y，返回2位的整数
*　获取月%m，值为1-12的整数

获取日%d，返回整数

*　获取时%H，值为0-23的整数

*　获取时%h，值为1-12的整数

*　获取分%i，值为0-59的整数

*　获取秒%s，值为0-59的整数

```
select date_format('2016-12-21','%Y %m %d');
```
当前日期current_date()
```
select current_date();
```
当前时间current_time()
```
select current_time();
```
当前日期时间now()
```
select now();
```