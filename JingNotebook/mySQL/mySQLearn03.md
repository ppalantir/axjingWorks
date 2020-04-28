# mySQL入门03 连接查询
## 前言
- 实体与实体之间有3种对应关系：一对一、一对多、多对多，这些关系也需要存储下来
- 在开发中需要对存储的数据进行一些处理，用到内置的一些函数
- 视图用于完成查询语句的封装
- 事务可以保证复杂的增删改操作有效

  - **问题???**
```
问：查询每个学生每个科目的分数
分析：学生姓名来源于students表，科目名称来源于subjects，分数来源于scores表，怎么将3个表放到一起查询，并将结果显示在同一个结果集中呢？
答：当查询结果来源于多张表时，需要使用连接查询
关键：找到表间的关系，当前的关系是
students表的id---scores表的stuid
subjects表的id---scores表的subid
则上面问题的答案是：
select students.sname,subjects.stitle,scores.score
from scores
inner join students on scores.stuid=students.id
inner join subjects on scores.subid=subjects.id;
结论：当需要对有关系的多张表进行查询时，需要使用连接join
```
- 连接查询
连接查询分类如下：
表A inner join 表B：表A与表B匹配的行会出现在结果中
表A left join 表B：表A与表B匹配的行会出现在结果中，外加表A中独有的数据，未对应的数据使用null填充
表A right join 表B：表A与表B匹配的行会出现在结果中，外加表B中独有的数据，未对应的数据使用null填充
在查询或条件中推荐使用“表名.列名”的语法
如果多个表中列名不重复可以省略“表名.”部分
如果表的名称太长，可以在表名后面使用' as 简写名'或' 简写名'，为表起个临时的简写名称

## 方法
1. 查询学生的姓名、平均分
```
select students.sname,avg(scores.score)
from scores
inner join students on scores.stuid=students.id
group by students.sname;
```

2. 查询男生的姓名、总分
```
select students.sname,avg(scores.score)
from scores
inner join students on scores.stuid=students.id
where students.gender=1
group by students.sname;
```
3. 查询科目的名称、平均分
```
select subjects.stitle,avg(scores.score)
from scores
inner join subjects on scores.subid=subjects.id
group by subjects.stitle;
```
4. 查询未删除科目的名称、最高分、平均分
```
select subjects.stitle,avg(scores.score),max(scores.score)
from scores
inner join subjects on scores.subid=subjects.id
where subjects.isdelete=0
group by subjects.stitle;
```