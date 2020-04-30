<!--
 * @Author: your name
 * @Date: 2020-04-30 20:20:37
 * @LastEditTime: 2020-04-30 20:32:59
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \axjingWorks\JingNotebook\mySQL\mySQLearn10.md
 -->
# mySQLearn09 python查 封装 用户登录
1. 查询一行数据
创建testSelectOne.py文件，查询一条学生信息
```python
import MySQLdb
try:
    conn=MySQLdb.connect(host='localhost',port=3306,db='test1',user='root',passwd='mysql',charset='utf8')
    cur=conn.cursor()
    cur.execute('select * from students where id=7')
    result=cur.fetchone()
    print result
    cur.close()
    conn.close()
except Exception,e:
    print e.message
```
2. 查询多行数据
创建testSelectMany.py文件，查询一条学生信息
```
import MySQLdb
try:
    conn=MySQLdb.connect(host='localhost',port=3306,db='test1',user='root',passwd='mysql',charset='utf8')
    cur=conn.cursor()
    cur.execute('select * from students')
    result=cur.fetchall()
    print result
    cur.close()
    conn.close()
except Exception,e:
    print e.message
```
3. 封装
观察前面的文件发现，除了sql语句及参数不同，其它语句都是一样的
创建MysqlHelper.py文件，定义类
```python
import MySQLdb
class MysqlHelper():
    def __init__(self,host,port,db,user,passwd,charset='utf8'):
        self.host=host
        self.port=port
        self.db=db
        self.user=user
        self.passwd=passwd
        self.charset=charset

    def connect(self):
        self.conn=MySQLdb.connect(host=self.host,port=self.port,db=self.db,user=self.user,passwd=self.passwd,charset=self.charset)
        self.cursor=self.conn.cursor()

    def close(self):
        self.cursor.close()
        self.conn.close()

    def get_one(self,sql,params=()):
        result=None
        try:
            self.connect()
            self.cursor.execute(sql, params)
            result = self.cursor.fetchone()
            self.close()
        except Exception, e:
            print e.message
        return result

    def get_all(self,sql,params=()):
        list=()
        try:
            self.connect()
            self.cursor.execute(sql,params)
            list=self.cursor.fetchall()
            self.close()
        except Exception,e:
            print e.message
        return list

    def insert(self,sql,params=()):
        return self.__edit(sql,params)

    def update(self, sql, params=()):
        return self.__edit(sql, params)

    def delete(self, sql, params=()):
        return self.__edit(sql, params)

    def __edit(self,sql,params):
        count=0
        try:
            self.connect()
            count=self.cursor.execute(sql,params)
            self.conn.commit()
            self.close()
        except Exception,e:
            print e.message
        return count
```
4. 添加
创建testInsertWrap.py文件，使用封装好的帮助类完成插入操作
```python
from MysqlHelper import *

sql='insert into students(sname,gender) values(%s,%s)'
sname=raw_input("请输入用户名：")
gender=raw_input("请输入性别，1为男，0为女")
params=[sname,bool(gender)]

mysqlHelper=MysqlHelper('localhost',3306,'test1','root','mysql')
count=mysqlHelper.insert(sql,params)
if count==1:
    print 'ok'
else:
    print 'error'
```
5. 查询一个
创建testGetOneWrap.py文件，使用封装好的帮助类完成查询最新一行数据操作
```python
from MysqlHelper import *

sql='select sname,gender from students order by id desc'

helper=MysqlHelper('localhost',3306,'test1','root','mysql')
one=helper.get_one(sql)
print one
```

## 用户登录
创建用户表userinfos
表结构如下
```
id
uname
upwd
isdelete
```
注意：需要对密码进行加密

如果使用md5加密，则密码包含32个字符

如果使用sha1加密，则密码包含40个字符，推荐使用这种方式
```
create table userinfos(
id int primary key auto_increment,
uname varchar(20),
upwd char(40),
isdelete bit default 0
);
```
6. 加入测试数据
插入如下数据，用户名为123,密码为123,这是sha1加密后的值
```
insert into userinfos values(0,'123','40bd001563085fc35165329ea1ff5c5ecbdbbeef',0);
```
7. 接收输入并验证
创建testLogin.py文件，引入hashlib模块、MysqlHelper模块
接收输入
根据用户名查询，如果未查到则提示用户名不存在
如果查到则匹配密码是否相等，如果相等则提示登录成功
如果不相等则提示密码错误
```python
from MysqlHelper import MysqlHelper
from hashlib import sha1

sname=raw_input("请输入用户名：")
spwd=raw_input("请输入密码:")

s1=sha1()
s1.update(spwd)
spwdSha1=s1.hexdigest()

sql="select upwd from userinfos where uname=%s"
params=[sname]

sqlhelper=MysqlHelper('localhost',3306,'test1','root','mysql')
userinfo=sqlhelper.get_one(sql,params)
if userinfo==None:
    print '用户名错误'
elif userinfo[0]==spwdSha1:
    print '登录成功'
else:
    print '密码错误'
```



**【欢迎关注、点赞、收藏、私信、交流】共同学习进步**
