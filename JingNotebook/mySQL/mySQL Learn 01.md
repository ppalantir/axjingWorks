# MySQL入门01 - 基本操作
* E-R模型
当前物理的数据库都是按照E-R模型进行设计的
E表示entry，实体
R表示relationship，关系
一个实体转换为数据库中的一个表
关系描述两个实体之间的对应规则，包括
一对一
一对多
多对多
关系转换为数据库表中的一个列 *在关系型数据库中一行就是一个对象
* 三范式
经过研究和对使用中问题的总结，对于设计数据库提出了一些规范，这些规范被称为范式
第一范式（1NF)：列不可拆分
第二范式（2NF)：唯一标识
第三范式（3NF)：引用主键
说明：后一个范式，都是在前一个范式的基础上建立的
* 数据完整性
一个数据库就是一个完整的业务单元，可以包含多张表，数据被存储在表中
在表中为了更加准确的存储数据，保证数据的正确有效，可以在创建表的时候，为表添加一些强制性的验证，包括数据字段的类型、约束
字段类型
在mysql中包含的数据类型很多，这里主要列出来常用的几种
```
数字：int,decimal
字符串：varchar,text
日期：datetime
布尔：bit
约束
主键primary key
非空not null
惟一unique
默认default
外键foreign key
```

# 准备工作 
## 安装
```
sudo apt-get install mysql-server
```
验证是否安装成功
```
sudo netstat -tap | grep mysql
```
## 管理服务
### 启动
```
service mysql start
```
### 停止
```
service mysql stop
```
### 重启
```
service mysql restart
```
## 允许远程连接
### 找到mysql配置文件并修改
```
sudo vi /etc/mysql/mysql.conf.d/mysqld.cnf
```
将bind-address=127.0.0.1注释
### 登录mysql，运行命令
```
grant all privileges on *.* to 'root'@'%' identified by 'mysql' with grant option;
flush privileges;
```
重启mysql

# 数据库的连接、创建、表创建、添加等
## 连接数据库
```
mysql -uroot -p
```
退出 quit/exit

## 操作
查看版本：select version();
显示当前时间：select now();
注意：在语句结尾要使用分号;
### 远程连接
```
mysql -h(ip地址) -uroot(用户名) -p(密码)
```
### 创建数据库
```
create database 数据库名称 charset=uuf8
```
### 删除数据库
```
drop database 数据库名称
```
### 切换数据库
```
use 数据库名称
```
### 查看当前数据库
```
select database();
```
### 显示所有数据库
```
show databases;
```
### 表操作

#### 显示所有表
```
show tables;
```
#### 创建表
```
create table 表名(
    #列及类型
    id int auto_increment primary key,
    sname varchar(10) not null,
    gender bit default 0,
    birthday datetime;
)
```

#### 修改表
```
alter table 表名 add|change|drop 列名 类型;
alter table students add isDelete bit default 0;
```
#### 删除表
```
drop table 表名；
物理删除，慎用
```
#### 查看表结构
```
desc 表名；
```
#### 更改名称
```
rename table 原表名 to 新表名；
```
#### 查看表的创建语句
```
show create table 表名；
```
#### 查询
```
* 全列插入：insert into 表名 value(...)
* 缺省插入：insert into 表名(列1,...) value(值1,...)
* 同时插入多条数据: inert into 表名(列1,...) value(值1,...),(值1,...),(值1,...),...
或者insert into 表名(...) value(值1,...),(值1,...),(值1,...),...
```
#### 修改
```
update 表名 set 列1=值1... where 条件
```

#### 删除
```
delete from 表名 where 条件
```

#### 逻辑删除，本质就是修改操作update
```
alter table students add isdelete bit default 0;
```
如果需要删除则
```
update students isdelete=1 where ...;
```
### 备份与恢复
#### 数据备份
进入超级管理员
```
sudo -s
```
进入mysql库目录
```cd /var/lib/mysql
```
运行mysqldump命令
```
mysqldump –uroot –p 数据库名 > ~/Desktop/备份文件.sql;
```
按提示输入mysql的密码
#### 数据恢复
连接mysqk，创建数据库

退出连接，执行如下命令
```
mysql -uroot –p 数据库名 < ~/Desktop/备份文件.sql
```
根据提示输入mysql密码