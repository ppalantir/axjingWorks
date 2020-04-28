# Tensorflow学习：

**重要：**
使用图（graph）来表示计算任务；
在被称之为会话（Session）的上下文（Context）中执行图；
使用tensor 表示数据
通过变量（Variable）维护状态；
使用feed和fatch可以为任意的操作（arbitrary operation）赋值或者从其中获取数据.
## 计算图
1. Tensorflow是一个编程系统，使用图来表示计算任务。途中的节点被称之为op(opteration的缩写).一个op获得0个或者多个Tensor，执行计算任务，产生0个或者多个Tensor.每个Tensor是一个类型化的多维数组
2. TensorFlow程序常常被组织成一个**构建阶段**和一个**执行阶段**，在构建阶段op的执行步骤被描述成一个图。在执行阶段，使用会话执行图中的op.
## 构建图
* 构建图的第一步是创建源op(source op).源op不需要任何的输入，例如常量(Constant).源op的输出被传递给其他的op做运算。

* python库中，op构造器的返回值代表被构造出来的op的输出， 这些返回值可以传递给其他的op构造器作为输入。

```python

import tensorflow as tf

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点# 加到默认图中.## 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)
```
默认途现在有三个节点，两个constant() op, 和一个matmul() op. 为了真正进行矩阵相乘运算，并得到矩阵的乘法的结果，你必须在会话里启动这个图

## 在一个会话中启动图
构造阶段完成后，才能启动图。启动图的第一步是创建一个Session对象，如果无任何创建参数，会话构造器将启动默认图.
```python

# 启动默认图.
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. # 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回# 矩阵乘法 op 的输出.## 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.# # 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.## 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print result
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
```

Session 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作.
```python
with tf.Session() as sess:
    result = sess.run([product])
    print result
```
在实现上，tensorflow将图像定义转换成分布式执行的操作，以充分利用可用的计算资源

## Tensor
TensorFlow程序使用Tensor数据结构来代表所有数据，计算图中，操作间传递的数据都是tensor.可以把TensorFlow 的tensor看做一个n维的数组或者列表。一个tensor包含一个静态的rank,和一个shape

TensorFlow用张量这种数据结构来表示所有的数据.你可以把一个张量想象成一个n维的数组或列表.一个张量有一个静态类型和动态类型的维数.张量可以在图中的节点之间流通.

* rank: 在TensorFlow系统中，张量的位数被描述为rank，但是张量的rank和矩阵的rank不同，并不是同一个概念.张量的rank(有时候是关于顺序、度数、n维)是张量维数的一个数量描述
如
```python
t = [[1,2,3], [4,5,6,], [7,8,9]]
```

你可以认为一个二阶张量就是我们平常所说的矩阵，一阶张量可以认为是一个向量.对于一个二阶张量你可以用语句t[i, j]来访问其中的任何元素.而对于三阶张量你可以用't[i, j, k]'来访问其中的任何元素.
* shape
Tensorflow文档中使用三种记号来方便的描绘张量的维度：rank,shape以及维数。下表表示他们之间的关系

|rank|shape|dim|
|---|---|---|
|0|[]|0-D|
|1|[D0]|1-D|
|2|[D0,D1]|1-D|
|3|[D0,D1,D]|3-D|


## 变量
Variables 变量维护了图的执行过程中的状态信息。例如：
```python

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图, 运行 opwith tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)
  # 打印 'state' 的初始值
  print sess.run(state)
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print sess.run(state)

输出:0# 1# 2# 3
```
## Fetch
为了取回操作的输出内容，可以在使用Session对象run()调用执行图时，传入一些tensor,这些tensor会帮助你取回结果。在之前的例子里，我们只取回了单个节点state,但是你也可以取回多个tensor:
```python

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session():
  result = sess.run([mul, intermed])
  print result

# 输出:# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
```

## Feed
* 计算图中引入了tensor,以常量或变量的形式存储。Tensorflow还提供了feed机制，改机制可以临时替代图中的任意操作中的tensor，可以对任意操作提供补丁，直接插入一个tensor.
* feed使用一个tensor值临时替换一个操作的输出结果。你可以提供feed数据作为run()的调用参数。feed只在调用它的方法内有效，方法结束，feed就会消失。最常见的用例是将某些特殊的操作指定为"feed"操作，标记的方法是使用tf.placeholder()为这些操作创建占位符

```python

input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

# 输出:# [array([ 14.], dtype=float32)]
```