# mySQL入门05 视图

## 子查询
### 查询支持嵌套使用
- 查询各学生的语文、数学、英语的成绩
```
select sname,
(select sco.score from scores sco inner join subjects sub on sco.subid=sub.id where sub.stitle='语文' and stuid=stu.id) as 语文,
(select sco.score from  scores sco inner join subjects sub on sco.subid=sub.id where sub.stitle='数学' and stuid=stu.id) as 数学,
(select sco.score from  scores sco inner join subjects sub on sco.subid=sub.id where sub.stitle='英语' and stuid=stu.id) as 英语
from students stu;
```

## 前言
- 对于复杂的查询，在多次使用后，维护是一件非常麻烦的事情
- 解决：定义视图
- 视图本质就是对查询的一个封装
- 定义视图
  
```
create view stuscore as 
select students.*,scores.score from scores
inner join students on scores.stuid=students.id;
```
视图的用途就是查询
```
select * from stuscore;
```

**【欢迎关注、点赞、收藏、私信、交流】**

