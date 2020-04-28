# mySQLearn07 事务
## 前言

- 当一个业务逻辑需要多个sql完成时，如果其中某条sql语句出错，则希望整个操作都退回
- 使用事务可以完成退回的功能，保证业务逻辑的正确性
- 事务四大特性(简称ACID)
    - 原子性(Atomicity)：事务中的全部操作在数据库中是不可分割的，要么全部完成，要么均不执行
    - 一致性(Consistency)：几个并行执行的事务，其执行结果必须与按某一顺序串行执行的结果相一致
    - 隔离性(Isolation)：事务的执行不受其他事务的干扰，事务执行的中间结果对其他事务必须是透明的
    - 持久性(Durability)：对于任意已提交事务，系统必须保证该事务对数据库的改变不被丢失，即使数据库出现故障
- 要求：表的类型必须是innodb或bdb类型，才可以对此表使用事务
1. 查看表的创建语句
```
show create table students;
```
2. 修改表的类型
```
alter table '表名' engine=innodb;
```
3. 事务语句
```
开启begin;
提交commit;
回滚rollback;
```
- 示例
步骤1：打开两个终端，连接mysql，使用同一个数据库，操作同一张表;

终端1：
```
select * from students;
```
---
终端2：
```
begin;
insert into students(sname) values('张飞');
```
步骤2

终端1：
```
select * from students;
```
步骤3

终端2：
```
commit;
```
---
终端1：
```
select * from students;
```
示例

步骤1：打开两个终端，连接mysql，使用同一个数据库，操作同一张表

终端1：
```
select * from students;
```
------------------------
终端2：
```
begin;
insert into students(sname) values('张飞');
```
步骤2

终端1：
```
select * from students;
```
步骤3

终端2：
```
rollback;
```
------------------------
终端1：
```
select * from students;
```
**【欢迎关注、点赞、收藏、私信、交流】**