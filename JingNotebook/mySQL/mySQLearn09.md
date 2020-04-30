<!--
 * @Author: your name
 * @Date: 2020-04-30 20:10:34
 * @LastEditTime: 2020-04-30 20:16:21
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \axjingWorks\JingNotebook\mySQL\mySQLearn09.md
 -->
# mySQLearn09 python增删改查
 增加
1. 创建testInsert.py文件，向学生表中插入一条数据
```python
#encoding=utf-8
import MySQLdb
try:
    conn=MySQLdb.connect(host='localhost',port=3306,db='test1',user='root',passwd='mysql',charset='utf8')
    cs1=conn.cursor()
    count=cs1.execute("insert into students(sname) values('张良')")
    print count
    conn.commit()
    cs1.close()
    conn.close()
except Exception,e:
    print e.message
```
2. 修改
创建testUpdate.py文件，修改学生表的一条数据
```python
import MySQLdb
try:
    conn=MySQLdb.connect(host='localhost',port=3306,db='test1',user='root',passwd='mysql',charset='utf8')
    cs1=conn.cursor()
    count=cs1.execute("update students set sname='刘邦' where id=6")
    print count
    conn.commit()
    cs1.close()
    conn.close()
except Exception,e:
    print e.message
```
3. 删除
创建testDelete.py文件，删除学生表的一条数据
```python
import MySQLdb
try:
    conn=MySQLdb.connect(host='localhost',port=3306,db='test1',user='root',passwd='mysql',charset='utf8')
    cs1=conn.cursor()
    count=cs1.execute("delete from students where id=6")
    print count
    conn.commit()
    cs1.close()
    conn.close()
except Exception,e:
    print e.message

```
4. sql语句参数化
创建testInsertParam.py文件，向学生表中插入一条数据
```python
import MySQLdb
try:
    conn=MySQLdb.connect(host='localhost',port=3306,db='test1',user='root',passwd='mysql',charset='utf8')
    cs1=conn.cursor()
    sname=raw_input("请输入学生姓名：")
    params=[sname]
    count=cs1.execute('insert into students(sname) values(%s)',params)
    print count
    conn.commit()
    cs1.close()
    conn.close()
except Exception,e:
    print e.message
```
5. 其它语句
cursor对象的execute()方法，也可以用于执行create table等语句
建议在开发之初，就创建好数据库表结构，不要在这里执行

**【欢迎关注、点赞、收藏、私信、交流】共同学习进步**