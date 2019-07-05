* ```axjingWorker```

  * 文件目录

    * addons 一些插件 
    * algorithm_note 算法基础，笔试高频题
    * scripts 文件预处理脚本
    * workspace 一些总结、心得
    * DeepLearning 手撸中......
    * ComputerVision 手撸中......
    * MachineLearning 手撸中......

***
# ML --->MachineLearning

## K-NN

## 决策树

## 随机深林

## Boosting

## SVM

## 神经网络

## 贝叶斯分类器

### 贝叶斯决策论

### 极大似然估计

### 朴素贝叶斯分类器

### 贝叶斯网络

### EM算法

## 主成分分析

### PAC算法

## 概率图模型

### 隐马尔科夫模型


### 马尔科夫随机场

### 条件随机场



***

# DL --->DeepLearning

## 基于梯度的学习

## 代价函数

## 输出单元

## logistic sigmoid与双曲线函数

## 万能近似性质

## 反向传播

### 微积分中的链式法则

### 反向传播算法


## 深度学习中的正则化

### 参数范数惩罚

#### $L^2$参数正则化

#### $L^1$正则化

### Dropout

## 模型中的优化

### 梯度下降

### 动量

### Nesterov动量

## 自适应学习率算法

### AdaGrad

### RMSProp

### Adam

## 二阶近似方法

### 牛顿法

### 共轭梯度

### BFGS

## 卷积网络

### 卷积运算

### 动机

### 池化

### 基本卷积函数的变体

### 结构化输出

### 数据类型

### 高效的卷积算法

## 蒙特卡洛方法

## 对数似然梯度

## 随机最大似然和对比散度

## 深度生成模型--玻尔兹曼机

```广泛使用的网络模型```\
## SqueezeNet:

《SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size》是希望降低网路的复杂度，即压缩模型、减少模型的大小，同时达到public网络的识别精度。但并不能提高网络的识别精度和速度。主要内容包括两点：（1）压缩层：利用1 x 1的卷积核，把N张输入特征图，压缩到M(N>M)张特征图，再输入下一层网络；（2）fire module网络单元。
二、网络设计思想
1、尽量用1x1的卷积核替代3x3的卷积核
尽可能使用1x1卷积核为主，因为1x1卷积核比3x3卷积核参数少了9倍。
2、引入Squeeze layer,尽量减少每一层的输入特征图数量
如对于3x3卷积层，参数的个数是(number of input channels) x (number of fiters) x (3x3).所以对于3x3的卷积核参数的减少策略有两种：filters个数（输出特征图个数），另一个方法是减少input channels个数。为此文献中引入了squeeze layers，利用1x1的卷积层，把本来是N张特征图的输入降M(M<N)张特征图，再做为输入。标准的一个卷积层：
filters shape = (3, 3, 64, 128)
3、延迟下采样操作
在Alexnet中，第一层卷积层stride=4，直接下采样了四倍，在一般的CNN中，一般的卷积层、池化层都会有下采样（stride>1）,甚至在前面几层网络的下采样比例会比较大，这样会导致最后几层的神经元的激活映射区域减少。为了提高精度可以设计下采样层延迟，延迟到后面几层进行下采样（这也是为什么本篇论文不能提高速度的原因，下采样越快后面的计算量就会大大减少）
三、fire module
根据前面的一些策略方案，文献设计了一个称之为fire module的单元，一个深度网络就是由n多个fire module串联在一起的网络，fire module如下图所示：

              图
结构解说：假设网络的输入时N张特征图
（1）首先通过1x1卷积层，把N张特征图FN压缩成M张特征图FM(M<N):
FM = fM(FN)
其中f表示卷积层的操作
（2）激活函数Relu层映射：FRELU = max(0, FM)
(3)以relu层的输出结果FRELU，分别构架1x1的卷积层，3x3的卷积层，然后把他们的输出特征图相连在一起：
Fout = [f1x1(FRELU), f3x3(FRELU)]


## U-Net:





## ShuffleNet:





## MobileNet:





## AlexNet:





## VGG:




## Inception:



## ResNet:







## FCN:






## R-CNN:




## Fast R-CNN:





## SSD:








## R-FCN:



## FPN:






## SPP:








## YOLOv1:






## YOLOv2:







## YOLOv3:






## GAN:


## 循环和递归网络（RNN）

### RNN

### LSTM




***
# CV基础 --->ComputerVision

## 灰度直方图

## 点运算&代数运算&几何运算

## 线性系统

### 卷积

### 多尺度表示

### 图像金字塔

## 傅里叶变换

## 滤波设计

## 采样数据处理

## 离散图像变换

## 小波变换

## 图像分割
```
1 目标分割(Target Segmentation)，任务是把目标对应的部分分割出来。
2 目标检测(Target Detection),检测到图片当中的目标的具体位置 
3 目标识别(Target Recognition),即是在所有的给定数据中，分类出哪一些sample是目标，哪一些不是。这个仅仅做一下分类任务。yes or no
4 目标追踪(Target Tracking),这个任务很重要的第一点是目标定位（Target Locating）

而且这个任务设计到的数据一般具有时间序列（Temporal Data）。常见的情况是首先Target被Identify以后，算法或者系统需要在接下来时序的数据中，快速并高效地对给定目标进行再定位。任务需要区别类似目标，需要避免不要的重复计算，充分利用好时序相关性（Temporal Correlation），并且需要对一些简单的变化Robust，必须旋转，遮盖，缩小放大，Motion Blur之类的线性或者非线性变化。
```

图像分割是计算机视觉研究中的一个经典难题，已经成为图像理解领域关注的一个热点，图像分割是图像分析的第一步，是计算机视觉的基础，是图像理解的重要组成部分，同时也是图像处理中最困难的问题之一。所谓图像分割是指根据灰度、彩色、空间纹理、几何形状等特征把图像划分成若干个互不相交的区域，使得这些特征在同一区域内表现出一致性或相似性，而在不同区域间表现出明显的不同。简单的说就是在一副图像中，把目标从背景中分离出来。对于灰度图像来说，区域内部的像素一般具有灰度相似性，而在区域的边界上一般具有灰度不连续性。关于图像分割技术，由于问题本身的重要性和困难性，从20世纪70年代起图像分割问题就吸引了很多研究人员为之付出了巨大的努力。虽然到目前为止，还不存在一个通用的完美的图像分割的方法，但是对于图像分割的一般性规律则基本上已经达成的共识，已经产生了相当多的研究成果和方法。

<center><font face="黑体" color=green size=6>传统分割方法</font></center>

利用数字图像处理、拓扑学、数学等方面的知识来进行图像分割的方法

### 阈值分割

阈值法的基本思想是基于图像的**灰度特征**来计算*一个或多个灰度阈值*，并将图像中每个像素的灰度值与阈值作比较，最后将像素根据比较结果分到合适的类别中。因此，该方法最为关键的一步就是按照某个准则函数来求解最佳灰度阈值。

使用范围：阈值法特别适用于**目标和背景占据不同灰度级范围**的图像。

图像若只有目标和背景两大类，那么只需要选取一个阈值进行分割，此方法成为单阈值分割；但是如果图像中有多个目标需要提取，单一阈值的分割就会出现作物，在这种情况下就需要选取多个阈值将每个目标分隔开，这种分割方法相应的成为多阈值分割。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144703.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图1 对数字的一种阈值分割方法。</font></center>
阀值分割方法的优缺点：
* 计算简单，效率较高；
* 只考虑像素点灰度值本身的特征，一般不考虑空间特征，因此对噪声比较敏感，鲁棒性不高。

由此可知，阈值分割方法的关键就在于阈值的选择。若将智能遗传算法应用在阀值筛选上，选取能最优分割图像的阀值，这可能是基于阀值分割的图像分割法的发展趋势。


### 基于梯度的图像分割


### 边缘检测和连接
基于边缘检测的图像分割算法试图通过检测包含不同区域的边缘来解决分割问题。它可以说是人们最先想到也是研究最多的方法之一。通常不同区域的边界上像素的灰度值变化比较剧烈，如果将图片从空间域通过傅里叶变换到频率域，边缘就对应着高频部分，这是一种非常简单的边缘检测算法。

边缘检测技术通常可以按照处理的技术分为串行边缘检测和并行边缘检测。串行边缘检测是要想确定当前像素点是否属于检测边缘上的一点，取决于先前像素的验证结果。并行边缘检测是一个像素点是否属于检测边缘高尚的一点取决于当前正在检测的像素点以及与该像素点的一些临近像素点。

最简单的边缘检测方法是并行微分算子法，它利用相邻区域的像素值不连续的性质，采用一阶或者二阶导数来检测边缘点。近年来还提出了基于曲面拟合的方法、基于边界曲线拟合的方法、基于反应-扩散方程的方法、串行边界查找、基于变形模型的方法。
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144728.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

边缘检测的优缺点：

（1）边缘定位准确；
（2）速度快；
（3）不能保证边缘的连续性和封闭性；
（4）在高细节区域存在大量的碎边缘，难以形成一个大区域，但是又不宜将高细节区域分成小碎片；

由于上述的（3）（4）两个难点，边缘检测只能产生边缘点，而非完整意义上的图像分割过程。这也就是说，在边缘点信息获取到之后还需要后续的处理或者其他相关算法相结合才能完成分割任务。

在以后的研究当中，用于提取初始边缘点的自适应阈值选取、用于图像的层次分割的更大区域的选取以及如何确认重要边缘以去除假边缘将变得非常重要。


### 基于区域的图像分割算法

基于区域的分割方法是以直接寻找区域为基础的分割技术，基于区域提取方法有两种基本形式：一种是区域生长，从单个像素出发，逐步合并以形成所需要的分割区域；另一种是从全局出发，逐步切割至所需的分割区域。

#### 区域增长
区域生长是从一组代表不同生长区域的种子像素开始，接下来将种子像素邻域里符合条件的像素合并到种子像素所代表的生长区域中，并将新添加的像素作为新的种子像素继续合并过程，直到找不到符合条件的新像素为止，该方法的关键是选择合适的初始种子像素以及合理的生长准则。

区域生长算法需要解决的三个问题：
  1 选择或确定一组能正确代表所需区域的种子像素；
  2 确定在生长过程中能将相邻像素包括进来的准则；
  3 指定让生长过程停止的条件或规则。

#### 区域分裂合并

区域生长是从某个或者某些像素点出发，最终得到整个区域，进而实现目标的提取。而分裂合并可以说是区域生长的逆过程，从整幅图像出发，不断的分裂得到各个子区域，然后再把前景区域合并，得到需要分割的前景目标，进而实现目标的提取。其实如果理解了上面的区域生长算法这个区域分裂合并算法就比较好理解啦。

四叉树分解法就是一种典型的区域分裂合并法，基本算法如下：
1 对于任一区域，如果$H(R_i)=False$就将其分裂成不重叠的四等分；
2 对相邻的两个区域$R_i$和$R_j$，它们也可以大小不同（即不在同一层），如果条件$H(R_iUR_j)=True$满足，就将它们合并起来；
3 如果进一步的分裂或合并都不可能，则结束。

其中R代表整个正方形图像区域，P代表逻辑词。
区域分裂合并算法优缺点：
（1）对复杂图像分割效果好；
（2）算法复杂，计算量大；
（3）分裂有可能破怪区域的边界。

在实际应用当中通常将区域生长算法和区域分裂合并算法结合使用，该类算法对某些复杂物体定义的复杂场景的分割或者对某些自然景物的分割等类似先验知识不足的图像分割效果较为理想。

### 二值图像处理

### 分割图像的结构化

### 基于聚类的分割方法

#### 分水岭算法

分水岭算法是一个非常好理解的算法，它根据分水岭的构成来考虑图像的分割，现实中我们可以想象成有山和湖的景象，那么一定是如下图的，水绕山山围水的景象。
分水岭分割方法，是一种基于拓扑理论的数学形态学的分割方法，其基本思想是把图像看作是测地学上的拓扑地貌，图像中每一点像素的灰度值表示该点的海拔高度，每一个局部极小值及其影响区域称为集水盆，而集水盆的边界则形成分水岭。分水岭的概念和形成可以通过模拟浸入过程来说明。在每一个局部极小值表面，刺穿一个小孔，然后把整个模型慢慢浸入水中，随着浸入的加深，每一个局部极小值的影响域慢慢向外扩展，在两个集水盆汇合处构筑大坝，即形成分水岭。
分水岭对微弱边缘具有良好的响应，图像中的噪声、物体表面细微的灰度变化都有可能产生过度分割的现象，但是这也同时能够保证得到封闭连续边缘。同时，分水岭算法得到的封闭的集水盆也为分析图像的区域特征提供了可能。
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144716.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图2</font></center>

#### K-means算法分割

### 基于小波分析和小波变换的图像分割算法

小波变换是近年来得到的广泛应用的数学工具，也是现在数字图像处理必学部分，它在时间域和频率域上都有量高的局部化性质，能将时域和频域统一于一体来研究信号。而且小波变换具有多尺度特性，能够在不同尺度上对信号进行分析，因此在图像分割方面的得到了应用，

二进小波变换具有检测二元函数的局部突变能力，因此可作为图像边缘检测工具。图像的边缘出现在图像局部灰度不连续处，对应于二进小波变换的模极大值点。通过检测小波变换模极大值点可以确定图像的边缘小波变换位于各个尺度上，而每个尺度上的小波变换都能提供一定的边缘信息，因此可进行多尺度边缘检测来得到比较理想的图像边缘。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144742.png" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

上图左图是传统的阈值分割方法，右边的图像就是利用小波变换的图像分割。可以看出右图分割得到的边缘更加准确和清晰

另外，将小波和其他方法结合起来处理图像分割的问题也得到了广泛研究，比如一种局部自适应阈值法就是将Hilbert图像扫描和小波相结合，从而获得了连续光滑的阈值曲线。

### 基于遗传算法的图像分割

遗传算法（Genetic Algorithms，简称GA）是1973年由美国教授Holland提出的，是一种借鉴生物界自然选择和自然遗传机制的随机化搜索算法。是仿生学在数学领域的应用。其基本思想是，模拟由一些基因串控制的生物群体的进化过程，把该过程的原理应用到搜索算法中，以提高寻优的速度和质量。此算法的搜索过程不直接作用在变量上，而是在参数集进行了编码的个体，这使得遗传算法可直接对结构对象（图像）进行操作。整个搜索过程是从一组解迭代到另一组解，采用同时处理群体中多个个体的方法，降低了陷入局部最优解的可能性，并易于并行化。搜索过程采用概率的变迁规则来指导搜索方向，而不采用确定性搜索规则，而且对搜索空间没有任何特殊要求（如连通性、凸性等），只利用适应性信息，不需要导数等其他辅助信息，适应范围广。

遗传算法擅长于全局搜索，但局部搜索能力不足，所以常把遗传算法和其他算法结合起来应用。将遗传算法运用到图像处理主要是考虑到遗传算法具有与问题领域无关且快速随机的搜索能力。其搜索从群体出发，具有潜在的并行性，可以进行多个个体的同时比较，能有效的加快图像处理的速度。但是遗传算法也有其缺点：搜索所使用的评价函数的设计、初始种群的选择有一定的依赖性等。要是能够结合一些启发算法进行改进且遗传算法的并行机制的潜力得到充分的利用，这是当前遗传算法在图像处理中的一个研究热点。

### 基于主动轮廓模型的分割算法

主动轮廓模型（active contours）是图像分割的一种重要方法，具有统一的开放式的描述形式，为图像分割技术的研究和创新提供了理想的框架。在实现主动轮廓模型时，可以灵活的选择约束力、初始轮廓和作用域等，以得到更佳的分割效果，所以主动轮廓模型方法受到越来越多的关注。

该方法是在给定图像中利用曲线演化来检测目标的一类方法，基于此可以得到精确的边缘信息。其基本思想是，先定义初始曲线C，然后根据图像数据得到能量函数，通过最小化能量函数来引发曲线变化，使其向目标边缘逐渐逼近，最终找到目标边缘。这种动态逼近方法所求得的边缘曲线具有封闭、光滑等优点。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144753.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

传统的主动轮廓模型大致分为参数主动轮廓模型和几何主动轮廓模型。参数主动轮廓模型将曲线或曲面的形变以参数化形式表达，Kass等人提出了经典的参数活动轮廓模型即“Snake”模型，其中Snake定义为能量极小化的样条曲线，它在来自曲线自身的内力和来自图像数据的外力的共同作用下移动到感兴趣的边缘，内力用于约束曲线形状，而外力则引导曲线到特征此边缘。参数主动轮廓模型的特点是将初始曲线置于目标区域附近，无需人为设定曲线的的演化是收缩或膨胀，其优点是能够与模型直接进行交互，且模型表达紧凑，实现速度快；其缺点是难以处理模型拓扑结构的变化。比如曲线的合并或分裂等。而使用水平集（level set）的几何活动轮廓方法恰好解决了这一问题。

<center><font face="黑体" color=green size=6>基于深度学习的分割算法</font></center>

### 基于特征编码（feature encoder based）

在特征提取领域中VGGnet和ResNet是两个非常有统治力的方法，接下来的一些篇幅会对这两个方法进行简短的介绍

#### VGGNet
由牛津大学计算机视觉组合和Google DeepMind公司研究员一起研发的深度卷积神经网络。它探索了卷积神经网络的深度和其性能之间的关系，通过反复的堆叠33的小型卷积核和22的最大池化层，成功的构建了16~19层深的卷积神经网络。VGGNet获得了ILSVRC 2014年比赛的亚军和定位项目的冠军，在top5上的错误率为7.5%。目前为止，VGGNet依然被用来提取图像的特征。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144850.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>
VGGNet的优缺点：
* 由于参数量主要集中在最后的三个FC当中，所以网络加深并不会带来参数爆炸的问题；
* 多个小核卷积层的感受野等同于一个大核卷积层（三个3x3等同于一个7x7）但是参数量远少于大核卷积层而且非线性操作也多于后者，使得其学习能力较强
* VGG由于层数多而且最后的三个全连接层参数众多，导致其占用了更多的内存（140M）

#### ResNet
随着深度学习的应用，各种深度学习模型随之出现，虽然在每年都会出现性能更好的新模型，但是对于前人工作的提升却不是那么明显，其中有重要问题就是深度学习网络在堆叠到一定深度的时候会出现梯度消失的现象，导致误差升高效果变差，后向传播时无法将梯度反馈到前面的网络层，使得前方的网络层的参数难以更新，训练效果变差。这个时候ResNet恰好站出来，成为深度学习发展历程中一个重要的转折点。

ResNet是由微软研究院的Kaiming He等四名华人提出，他们通过自己提出的ResNet Unit成功训练出来152层的神经网络并在ILSVRC2015比赛中斩获冠军。ResNet语义分割领域最受欢迎且最广泛运用的神经网络.ResNet的核心思想就是在网络中引入恒等映射，允许原始输入信息直接传到后面的层中，在学习过程中可以只学习上一个网络输出的残差（F(x)），因此ResNet又叫做残差网络。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705155354.png" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

使用到ResNet的分割模型：

Efficient Neural Network（ENet）：该网络类似于ResNet的bottleNeck方法；
ResNet-38：该网络在训练or测试阶段增加并移除了一些层，是一种浅层网络，它的结构是ResNet+FCN；
full-resolution residual network(FRRN)：FRRN网络具有和ResNet相同优越的训练特性，它由残差流和池化流两个处理流组成；
AdapNey：根据ResNet-50的网络进行改进，让原本的ResNet网络能够在更短的时间内学习到更多高分辨率的特征；
……

ResNet的优缺点：
1）引入了全新的网络结构（残差学习模块），形成了新的网络结构，可以使网络尽可能地加深；
2）使得前馈/反馈传播算法能够顺利进行，结构更加简单；
3）恒等映射地增加基本上不会降低网络的性能；
4）建设性地解决了网络训练的越深，误差升高，梯度消失越明显的问题；
5）由于ResNet搭建的层数众多，所以需要的训练时间也比平常网络要长。

### 基于区域选择（regional proposal based）
Regional proposal 在计算机视觉领域是一个非常常用的算法，尤其是在目标检测领域。其核心思想就是检测颜色空间和相似矩阵，根据这些来检测待检测的区域。然后根据检测结果可以进行分类预测。

在语义分割领域，基于区域选择的几个算法主要是由前人的有关于目标检测的工作渐渐延伸到语义分割的领域的，接下来小编将逐步介绍其个中关系。

#### Stage Ⅰ：R-CNN

伯克利大学的Girshick教授等人共同提出了首个在目标检测方向应用的深度学习模型：Region-based Convolutional Neural Network（R-CNN）。该网络模型如下图所示，其主要流程为：先使用selective search算法提取2000个候选框，然后通过卷积网络对候选框进行串行的特征提取，再根据提取的特征使用SVM对候选框进行分类预测，最后使用回归方法对区域框进行修正。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144921.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>
R-CNN的优缺点：

是首个开创性地将深度神经网络应用到目标检测的算法；
使用Bounding Box Regression对目标检测的框进行调整；
由于进行特征提取时是串行，处理耗时过长；
Selective search算法在提取每一个region时需要2s的时间，浪费大量时间

#### Stage Ⅱ：Fast R-CNN
由于R-CNN的效率太低，2015年由Ross等学者提出了它的改进版本：Fast R-CNN。其网络结构图如下图所示（从提取特征开始，略掉了region的选择）Fast R-CNN在传统的R-CNN模型上有所改进的地方是它是直接使用一个神经网络对整个图像进行特征提取，就省去了串行提取特征的时间；接着使用一个RoI Pooling Layer在全图的特征图上摘取每一个RoI对应的特征，再通过FC进行分类和包围框的修正。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144927.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

Fast R-CNN的优缺点

节省了串行提取特征的时间；
除了selective search以外的其它所有模块都可以合在一起训练；
最耗时间的selective search算法依然存在。

#### Stage Ⅲ：Faster R-CNN

2016年提出的Faster R-CNN可以说有了突破性的进展（虽然还是目标检测哈哈哈），因为它改变了它的前辈们最耗时最致命的部位：selective search算法。它将selective search算法替换成为RPN，使用RPN网络进行region的选取，将2s的时间降低到10ms，其网络结构如下图所示：

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144932.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

Faster R-CNN优缺点：

使用RPN替换了耗时的selective search算法，对整个网络结构有了突破性的优化；
Faster R-CNN中使用的RPN和selective search比起来虽然速度更快，但是精度和selective search相比稍有不及，如果更注重速度而不是精度的话完全可以只使用RPN；

#### Stage Ⅳ：Mask R-CNN

Mask R-CNN（终于到分割了！）是何恺明大神团队提出的一个基于Faster R-CNN模型的一种新型的分割模型，此论文斩获ICCV 2017的最佳论文，在Mask R-CNN的工作中，它主要完成了三件事情：目标检测，目标分类，像素级分割。
恺明大神是在Faster R-CNN的结构基础上加上了Mask预测分支，并且改良了ROI Pooling，提出了ROI Align。其网络结构真容就如下图所示啦：

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144938.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

Mask R-CNN的优缺点：

引入了预测用的Mask-Head，以像素到像素的方式来预测分割掩膜，并且效果很好；
用ROI Align替代了ROI Pooling，去除了RoI Pooling的粗量化，使得提取的特征与输入良好对齐；
分类框与预测掩膜共享评价函数，虽然大多数时间影响不大，但是有的时候会对分割结果有所干扰。

#### Mask Scoring R-CNN

最后要提出的是2019年CVPR的oral，来自华中科技大学的研究生黄钊金同学提出的

MS R-CNN，这篇文章的提出主要是对上文所说的Mask R-CNN的一点点缺点进行了修正。他的网络结构也是在Mask R-CNN的网络基础上做了一点小小的改进，添加了Mask-IoU。

黄同学在文章中提到：恺明大神的Mask R-CNN已经很好啦！但是有个小毛病，就是评价函数只对目标检测的候选框进行打分，而不是分割模板（就是上文提到的优缺点中最后一点），所以会出现分割模板效果很差但是打分很高的情况。所以黄同学增加了对模板进行打分的MaskIoU Head，并且最终的分割结果在COCO数据集上超越了恺明大神，下面就是MS R-CNN的网络结构啦~

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144946.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

MS R-CNN的优缺点：

优化了Mask R-CNN中的信息传播，提高了生成预测模板的质量；
未经大批量训练的情况下，就拿下了COCO 2017挑战赛实例分割任务冠军；
要说缺点的话。。应该就是整个网络有些庞大，一方面需要ResNet当作主干网络，另一方面需要其它各种Head共同承担各种任务。


### 基于RNN的图像分割算法

Recurrent neural networks（RNNs）除了在手写和语音识别上表现出色外，在解决计算机视觉的任务上也表现不俗，在本篇文章中我们就将要介绍RNN在2D图像处理上的一些应用，其中也包括介绍使用到它的结构或者思想的一些模型。

RNN是由Long-Short-Term Memory（LSTM）块组成的网络，RNN来自序列数据的长期学习的能力以及随着序列保存记忆的能力使其在许多计算机视觉的任务中游刃有余，其中也包括语义分割以及数据标注的任务。接下来的部分我们将介绍几个使用到RNN结构的用于分割的网络结构模型：

#### ReSeg模型
ReSeg可能不被许多人所熟知，在百度上搜索出的相关说明与解析也不多，但是这是一个很有效的语义分割方法。众所周知，FCN可谓是图像分割领域的开山作，而RegNet的作者则在自己的文章中大胆的提出了FCN的不足：没有考虑到局部或者全局的上下文依赖关系，而在语义分割中这种依赖关系是非常有用的。所以在ReSeg中作者使用RNN去检索上下文信息，以此作为分割的一部分依据。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705144954.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

该结构的核心就是Recurrent Layer，它由多个RNN组合在一起，捕获输入数据的局部和全局空间结构。

优缺点：
充分考虑了上下文信息关系；
使用了中值频率平衡，它通过类的中位数(在训练集上计算)和每个类的频率之间的比值来重新加权类的预测。这就增加了低频率类的分数，这是一个更有噪声的分割掩码的代价，因为被低估的类的概率被高估了，并且可能导致在输出分割掩码中错误分类的像素增加。

#### MDRNNs（Multi-Dimensional Recurrent Neural Networks）模型
传统的RNN在一维序列学习问题上有着很好的表现，比如演讲（speech）和在线手写识别。但是 在多为问题中应用却并不到位。MDRNNs在一定程度上将RNN拓展到多维空间领域，使之在图像处理、视频处理等领域上也能有所表现。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145010.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

该论文的基本思想是：将单个递归连接替换为多个递归连接，相应可以在一定程度上解决时间随数据样本的增加呈指数增长的问题。以下就是该论文提出的两个前向反馈和反向反馈的算法。

### 基于上采样、反卷积的分割方法

卷积神经网络在进行采样的时候会丢失部分细节信息，这样的目的是得到更具特征的价值。但是这个过程是不可逆的，有的时候会导致后面进行操作的时候图像的分辨率太低，出现细节丢失等问题。因此我们通过上采样在一定程度上可以不全一些丢失的信息，从而得到更加准确的分割边界。

接下来介绍几个非常著名的分割模型：
#### FCN(Fully Convolutional Network)
FCN！在图像分割领域已然成为一个业界标杆，大多数的分割方法多多少少都会利用到FCN或者其中的一部分，比如前面我们讲过的Mask R-CNN。
在FCN当中的反卷积-升采样结构中，图片会先进性上采样（扩大像素）；再进行卷积——通过学习获得权值。FCN的网络结构如下图所示：
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145020.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

优缺点：

FCN对图像进行了像素级的分类，从而解决了语义级别的图像分割问题；
FCN可以接受任意尺寸的输入图像，可以保留下原始输入图像中的空间信息；
得到的结果由于上采样的原因比较模糊和平滑，对图像中的细节不敏感；
对各个像素分别进行分类，没有充分考虑像素与像素的关系，缺乏空间一致性。

#### SetNet

SegNet是剑桥提出的旨在解决自动驾驶或者智能机器人的图像语义分割深度网络，SegNet基于FCN，与FCN的思路十分相似，只是其编码-解码器和FCN的稍有不同，其解码器中使用去池化对特征图进行上采样，并在分各种保持高频细节的完整性；而编码器不使用全连接层，因此是拥有较少参数的轻量级网络：
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145031.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

SetNet的优缺点：

保存了高频部分的完整性；
网络不笨重，参数少，较为轻便；
对于分类的边界位置置信度较低；
对于难以分辨的类别，例如人与自行车，两者如果有相互重叠，不确定性会增加。

以上两种网络结构就是基于反卷积/上采样的分割方法，当然其中最最最重要的就是FCN了，哪怕是后面大名鼎鼎的SegNet也是基于FCN架构的，而且FCN可谓是语义分割领域中开创级别的网络结构。

### 基于提高分辨率的分割方法

在这一个模块中我们主要给大家介绍一下基于提升特征分辨率的图像分割的方法。换一种说法其实可以说是恢复在深度卷积神经网络中下降的分辨率，从而获取更多的上下文信息。这一系列我将给大家介绍的是Google提出的DeepLab 。
DeepLab是结合了深度卷积神经网络和概率图模型的方法，应用在语义分割的任务上，目的是做逐像素分类，其先进性体现在DenseCRFs（概率图模型）和DCNN的结合。是将每个像素视为CRF节点，利用远程依赖关系并使用CRF推理直接优化DCNN的损失函数。
在图像分割领域，FCN的一个众所周知的操作就是平滑以后再填充，就是先进行卷积再进行pooling,这样在降低图像尺寸的同时增大感受野，但是在先减小图片尺寸（卷积）再增大尺寸（上采样）的过程中一定有一些信息损失掉了，所以这里就有可以提高的空间。
接下来我要介绍的是DeepLab网络的一大亮点：Dilated/Atrous Convolution，它使用的采样方式是带有空洞的采样。在VGG16中使用不同采样率的空洞卷积，可以明确控制网络的感受野。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145041.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图a对应3x3的1-dilated conv，它和普通的卷积操作是相同的；图b对应3x3的2-dilated conv，事迹卷积核的尺寸还是3x3（红点），但是空洞为1，其感受野能够达到7x7；图c对应3x3的4-dilated conv，其感受野已经达到了15x15.写到这里相信大家已经明白，在使用空洞卷积的情况下，加大了感受野，使每个卷积输出都包含了较大范围的信息。</font></center>

这样就解决了DCNN的几个关于分辨率的问题：
1）内部数据结构丢失；空间曾计划信息丢失；
2）小物体信息无法重建；

当然空洞卷积也存在一定的问题，它的问题主要体现在以下两方面：
1）网格效应
加入我们仅仅多次叠加dilation rate 2的 3x3 的卷积核则会出现以下问题

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145051.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

由此发现卷积核并不连续，也就是说并不是所有的像素都用来计算了，这样会丧失信息的连续性；


2）小物体信息处理不当

我们从空洞卷积的设计背景来看可以推测出它是设计来获取long-ranged information。然而空洞步频选取得大获取只有利于大物体得分割，而对于小物体的分割可能并没有好处。所以如何处理好不同大小物体之间的关系也是设计好空洞卷积网络的关键。

### 基于特征增强的分割方法

基于特征增强的分割方法包括：提取多尺度特征或者从一系列嵌套的区域中提取特征。在图像分割的深度网络中，CNN经常应用在图像的小方块上，通常称为以每个像素为中心的固定大小的卷积核，通过观察其周围的小区域来标记每个像素的分类。在图像分割领域，能够覆盖到更大部分的上下文信息的深度网络通常在分割的结果上更加出色，当然这也伴随着更高的计算代价。多尺度特征提取的方法就由此引进。
在这一模块中我先给大家介绍一个叫做SLIC，全称为simple linear iterative cluster的生成超像素的算法。
首先我们要明确一个概念：啥是超像素？其实这个比较容易理解，就像上面说的“小方块”一样，我们平常处理图像的最小单位就是像素了，这就是像素级（pixel-level）；而把像素级的图像划分成为区域级（district-level）的图像，把区域当成是最基本的处理单元，这就是超像素啦。
算法大致思想是这样的，将图像从RGB颜色空间转换到CIE-Lab颜色空间，对应每个像素的（L，a，b）颜色值和（x，y）坐标组成一个5维向量V[l, a, b, x, y],两个像素的相似性即可由它们的向量距离来度量，距离越大，相似性越小。
算法首先生成K个种子点，然后在每个种子点的周围空间里搜索距离该种子点最近的若干像素，将他们归为与该种子点一类，直到所有像素点都归类完毕。然后计算这K个超像素里所有像素点的平均向量值，重新得到K个聚类中心，然后再以这K个中心去搜索其周围与其最为相似的若干像素，所有像素都归类完后重新得到K个超像素，更新聚类中心，再次迭代，如此反复直到收敛。
有点像聚类的K-Means算法，最终会得到K个超像素。
Mostahabi等人提出的一种前向传播的分类方法叫做Zoom-Out就使用了SLIC的算法，它从多个不同的级别提取特征：局部级别：超像素本身；远距离级别：能够包好整个目标的区域；全局级别：整个场景。这样综合考虑多尺度的特征对于像素或者超像素的分类以及分割来说都是很有意义的。
接下来的部分我将给大家介绍另一种完整的分割网络：PSPNet：Pyramid Scene Parsing Network
论文提出在场景分割是，大多数的模型会使用FCN的架构，但是FCN在场景之间的关系和全局信息的处理能力存在问题，其典型问题有：1.上下文推断能力不强；2.标签之间的关系处理不好；3.模型可能会忽略小的东西。
本文提出了一个具有层次全局优先级，包含不同子区域时间的不同尺度的信息，称之为金字塔池化模块。
该模块融合了4种不同金字塔尺度的特征，第一行红色是最粗糙的特征–全局池化生成单个bin输出，后面三行是不同尺度的池化特征。为了保证全局特征的权重，如果金字塔共有N个级别，则在每个级别后使用1×1 1×11×1的卷积将对于级别通道降为原本的1/N。再通过双线性插值获得未池化前的大小，最终concat到一起。其结构如下图：

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145102.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

最终结果就是，在融合不同尺度的feature后，达到了语义和细节的融合，模型的性能表现提升很大，作者在很多数据集上都做过训练，最终结果是在MS-COCO数据集上预训练过的效果最好。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145110.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

为了捕捉多尺度特征，高层特征包含了更多的语义和更少的位置信息。结合多分辨率图像和多尺度特征描述符的优点，在不丢失分辨率的情况下提取图像中的全局和局部信息，这样就能在一定程度上提升网络的性能。

#### 使用CRF/MRF的方法

MRF全称是Marcov Random Field，马尔可夫随机场，其实说起来笔者在刚读硕士的时候有一次就有同学在汇报中提到了隐马尔可夫、马尔可夫链啥的，当时还啥都不懂，小白一枚（现在是准小白hiahia），觉得马尔可夫这个名字贼帅，后来才慢慢了解什么马尔科夫链呀，马尔可夫随机场，并且在接触到图像分割了以后就对马尔科夫随机场有了更多的了解。
MRF其实是一种基于统计的图像分割算法，马尔可夫模型是指一组事件的集合，在这个集合中，事件逐个发生，并且下一刻事件的发生只由当前发生的事件决定，而与再之前的状态没有关系。而马尔可夫随机场，就是具有马尔可夫模型特性的随机场，就是场中任何区域都只与其临近区域相关，与其他地方的区域无关，那么这些区域里元素（图像中可以是像素）的集合就是一个马尔可夫随机场。
CRF的全称是Conditional Random Field，条件随机场其实是一种特殊的马尔可夫随机场，只不过是它是一种给定了一组输入随机变量X的条件下另一组输出随机变量Y的马尔可夫随机场，它的特点是埃及设输出随机变量构成马尔可夫随机场，可以看作是最大熵马尔可夫模型在标注问题上的推广。
在图像分割领域，运用CRF比较出名的一个模型就是全连接条件随机场（DenseCRF），接下来我们将花费一些篇幅来简单介绍一下。
CRF在运行中会有一个问题就是它只对相邻节点进行操作，这样会损失一些上下文信息，而全连接条件随机场是对所有节点进行操作，这样就能获取尽可能多的临近点信息，从而获得更加精准的分割结果。
在Fully connected CRF中，吉布斯能量可以写作：
\[{\rm{E}}(x) = \sum\limits_i {{\theta _i}({x_i}) + \sum\limits_{ij} {{\theta _{ij}}({x_i},{y_j})} } \]
其中k(m)为高斯核，写作：
\[{{\rm{k}}^m}({f_i},{f_j}) = {\omega _1}\exp ( - \frac{{{{\left\| {{p_i} + {p_j}} \right\|}^2}}}{{2\sigma _\alpha ^2}} - \frac{{{{\left\| {{I_i} + {I_j}} \right\|}^2}}}{{2\sigma _\beta ^2}}) + {\omega _2}\exp ( - \frac{{{{\left\| {{p_i} + {p_j}} \right\|}^2}}}{{2\sigma _\gamma ^2}})\]
该模型的一元势能包含了图像的形状，纹理，颜色和位置，二元势能使用了对比度敏感的的双核势能，CRF的二元势函数一般是描述像素点与像素点之间的关系，鼓励相似像素分配相同的标签，而相差较大的像素分配不同标签，而这个“距离”的定义与颜色值和实际相对距离有关，这样CRF能够使图像尽量在边界处分割。全连接CRF模型的不同就在于其二元势函数描述的是每一个像素与其他所有像素的关系，使用该模型在图像中的所有像素对上建立点对势能从而实现极大地细化和分割。
在分割结果上我们可以看看如下的结果图：
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190705145133.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图</font></center>

可以看到它在精细边缘的分割比平常的分割方法要出色得多，而且文章中使用了另一种优化算法，使得本来需要及其大量运算的全连接条件随机场也能在很短的时间里给出不错的分割结果。

至于其优缺点，我觉得可以总结为以下几方面：
在精细部位的分割非常优秀；
充分考虑了像素点或者图片区域之间的上下文关系；
在粗略的分割中可能会消耗不必要的算力；
可以用来恢复细致的局部结构，但是相应的需要较高的代价。

## 图像跟踪

### 基于卡尔曼滤波器的线性动态模型跟踪

## 物体测量

### 尺寸测量

### 形状分析

### 纹理分析

### 曲线和曲面拟合

## 局部图像特征
### HOG:



### SIFT:


## GIST:






## LBP:






## Harr:

## 霍夫变换

## 动态规划

## 隐马尔科夫模型

***

# 线性代数

## 线性相关

## 生成子空间

## 范数

## 特征分解

## 奇异值分解

***

# 概率与信息论

## 随机变量

## 概率分布

### 离散型变量和概率质量函数

### 连续型变量和概率密度函数

## 边缘概率

## 条件概率

## 条件概率的链式法则

## 独立性和条件独立性

## 期望、方差和协方差

## 常用概率分布
### Bernoulli分布

### Multinoulli分布

### 高斯分布

### 指数分布

### Laplace分布

### Dirac分布和经验分布

## 常用函数的性质

## 贝叶斯规则

## 连续变量的技术细节

## 信息论

## 结构化概率模型

## 梯度优化方法
## Jacobian矩阵

## Hessian矩阵

## 最小二乘法

## 最大似然估计


***
# DL在医学图像处理中的应用
0. 引言

医学图像处理的对象是各种不同成像机理的医学影像，临床广泛使用的医学成像种类主要有X-射线成像 （X-CT）、核磁共振成像（MRI）、核医学成像（NMI）和超声波成像（UI）四类。在目前的影像医疗诊断中，主要是通过观察一组二维切片图象去发现病变体，这往往需要借助医生的经验来判定。利用计算机图象处理技术对二维切片图象进行分析和处理，实现对人体器官、软组织和病变体的分割提取、三维重建和三维显示，可以辅助医生对病变体及其它感兴趣的区域进行定性甚至 定量的分析，从而大大提高医疗诊断的准确性和可靠性；在医疗教学、手术规划、手术仿真及各种医学研究中也能起重要的辅助作用[1,2]。目前，医学图像处理主要集中表现在病变检测、图像分割、图像配准及图像融合四个方面。

用深度学习方法进行数据分析呈现快速增长趋势，称为2013年的10项突破性技术之一。深度学习是人工神经网络的改进，由更多层组成，允许更高层次包含更多抽象信息来进行数据预测。迄今为止，它已成为计算机视觉领域中领先的机器学习工具，深度神经网络学习自动从原始数据（图像）获得的中级和高级抽象特征。最近的结果表明，从CNN中提取的信息在自然图像中的对目标识别和定位方面非常有效。世界各地的医学图像处理机构已经迅速进入该领域，并将CNN和其它深度学习方法应用于各种医学图像分析。

在医学成像中，疾病的准确诊断和评估取决于医学图像的采集和图像解释。近年来，图像采集已经得到了显着改善，设备以更快的速率和更高的分辨率采集数据。然而，图像解释过程，最近才开始受益于计算机技术。对医学图像的解释大多数都是由医生进行的，然而医学图像解释受到医生主观性、医生巨大差异认知和疲劳的限制。

用于图像处理的典型CNN架构由一系列卷积网络组成，其中包含有一系列数据缩减即池化层。与人脑中的低级视觉处理一样，卷积网络检测提取图像特征，例如可能表示直边的线或圆（例如器官检测）或圆圈（结肠息肉检测），然后是更高阶的特征，例如局部和全局形状和纹理特征提取[3]。CNN的输出通常是一个或多个概率或种类标签。

CNN是高度可并行化的算法。与单核的CPU处理相比，今天使用的图形处理单元（GPU）计算机芯片实现了大幅加速（大约40倍）。在医学图像处理中，GPU首先被引入用于分割和重建，然后用于机器学习。由于CNN的新变种的发展以及针对现代GPU优化的高效并行网络框架的出现，深度神经网络吸引了商业兴趣。从头开始训练深度CNN是一项挑战[4]。首先，CNN需要大量标记的训练数据，这一要求在专家注释昂贵且疾病稀缺的医学领域中可能难以满足。其次，训练深度CNN需要大量的计算和内存资源，否则训练过程将是非常耗时。第三，深度CNN训练过程中由于过度拟合和收敛问题而复杂化，这通常需要对网络的框架结构或学习参数进行重复调整，以确保所有层都以相当的速度学习[5]。鉴于这些困难，一些新的学习方案，称为“迁移学习”和“微调”，被证明可以解决上述问题从而越来越受欢迎。

1. 病变检测

计算机辅助检测（CAD）是医学图像分析的有待完善的领域，并且非常适合引入深度学习。在CAD 的标准方法中，一般通过监督方法或经典图像处理技术（如过滤和数学形态学）检测候选病变位置。**病变位置检测是分阶段的**，并且通常由大量手工制作的特征描述。将分类器用于特征向量映射到候选区来检测实际病变的概率。采用深度学习的直接方式是训练CNN操作一组以图像为中心的图像数据候选病变。Setio等在3D胸部CT扫描中检测肺结节，并在九个不同方向上提取以这些候选者为中心的2D贴片[6]，使用不同CNN的组合来对每个候选者进行分类，CAD系统结构如图1所示。根据检测结果显示，与先前公布的用于相同任务的经典CAD系统相比略有改进。罗斯等人应用CNN改进三种现有的CAD系统，用于检测CT成像中的结肠息肉，硬化性脊柱变形和淋巴结肿大[7]。他们还在三个正交方向上使用先前开发的候选检测器和2D贴片，以及多达100个随机旋转的视图。随机旋转的“2.5D”视图是从原始3D数据分解图像的方法。采用CNN对这些2.5D视图图像检测然后汇总，来提高检测的准确率。对于使用CNN的三个CAD系统，病变检测的准确率度提高了13-34％，而使用非深度学习分类器（例如支持向量机）几乎不可能实现这种程度的提升。早在1996年，Sahiner等人就已将CNN应用于医学图像处理。从乳房X线照片中提取肿块或正常组织的ROI。 CNN由输入层，两个隐藏层和输出层组成，并用于反向传播。在“GPU时代”以前，训练时间被描述为“计算密集型”，但没有给出任何时间。1993年，CNN应用于肺结节检测；1995年CNN用于检测乳腺摄影中的微钙化。
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211707.jpg" width = "480" height = "300" /></div>
<font face="黑体" color=gray size=1>图1. CAD系统概述。（a）从立方体的九个对称平面中提取的二维斑块的示例。候选者位于贴片的中心，边界框为50 50 mm和64 64 px。（b）通过合并专门为固体，亚固体和大结节设计的探测器的输出来检测候选人。误报减少阶段是作为多个ConvNets的组合实现的。每个ConvNets流处理从特定视图中提取的2-D补丁。（c）融合每个ConvNet流输出的不同方法。 灰色和橙色框表示来自第一个完全连接的层和结节分类输出的连接神经元。 使用完全连接的层与softmax或固定组合器（产品规则）组合神经元。（a）使用体积对象的九个视图提取二维补丁。（b）拟议系统的示意图。（c）融合方法 </font>

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211731.jpg" width = "480" height = "300" /></div></center>
<center><font face="黑体" color=gray size=1>图2.结肠息肉的检测：不同息肉大小的FROC曲线，使用792测试CT结肠成像患者的随机视图ConvNet观察。 </font></center>

2. 图像分割

医学图像分割就是一个根据区域间的相似或不同把图像分割成若干区域的过程。目前，主要以各种细胞、组织与器官的图像作为处理的对象。传统的图像分割技术有基于区域的分割方法和基于边界的分割方法，前者依赖于图像的空间局部特征，如灰度、纹理及其它象素统计特性的均匀性等，后者主要是利用梯度信息确定目标的边界。结合特定的理论工具，图象分割技术有了更进一步的发展。比如基于三维可视化系统结合FastMarching算法和Watershed 变换的医学图象分割方法，能得到快速、准确的分割结果[8]。
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211739.jpg" width = "480" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图3.Watershed 变换的医学图象分割方法 </font></center>

近年来，随着其它新兴学科的发展，产生了一些全新的图像分割技术。如**基于统计学的方法、基于模糊理论的方法、基于神经网络的方法、基于小波分析的方法、基于模型的snake 模型(动态轮廓模型)、组合优化模型**等方法。虽然不断有新的分割方法被提出，但结果都不是很理想。目前研究的热点是一种基于知识的分割方法，即通过某种手段将一些先验的知识导入分割过程中，从而约束计算机的分割过程，使得分割结果控制在我们所能认识的范围内而不至于太离谱。比如在肝内部肿块与正常肝灰度值差别很大时，不至于将肿块与正常肝看成 2 个独立的组织。

医学图像分割方法的研究具有如下显著特点：现有任何一种单独的图像分割算法都难以对一般图像取得比较满意的结果，要更加注重多种分割算法的有效结合；由于人体解剖结构的复杂性和功能的系统性，虽然已有研究通过医学图像的自动分割区分出所需的器官、组织或找到病变区的方法，但目前现成的软件包一般无法完成全自动的分割，尚需要解剖学方面的人工干预[9]。在目前无法完全由计算机来完成图像分割任务的情况下，人机交互式分割方法逐渐成为研究重点；新的分割方法的研究主要以自动、精确、快速、自适应和鲁棒性等几个方向作为研究目标，经典分割技术与现代分割技术的综合利用(集成技术)是今后医学图像分割技术的发展方向[10,11]。

利用2891次心脏超声检查的数据集，Ghesu等结合深度学习和边缘空间学习进行医学图像检测和分割[12]。“大参数空间的有效探索”和在深度网络中实施稀疏性的方法相结合，提高了计算效率，并且与同一组发布的参考方法相比，平均分割误差减少了13.5％，八位患者的检测结果如图4所示。Brosch等人利用MRI图像上研究多发性硬化脑病变分割的问题。开发了一种3D深度卷积编码器网络，它结合了卷积和反卷积[13]，图5.增加网络深度对病变的分割性能的影响。卷积网络学习了更高级别的特征，并且反卷积网络预进行像素级别分割。将网络应用于两个公开的数据集和一个临床试验数据集，与5种公开方法进行了比较，展现了最好的方法。Pereira等人的研究中对MRI上的脑肿瘤分割进行了研究，使用更深层的架构，数据归一化和数据增强技巧[14]。将不同的CNN架构用于肿瘤，该方法分别对疑似肿瘤的图像增强和核心区域进行分割。在2013年的公共挑战数据集上获得了最高成绩。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/Snipaste_2019-07-04_22-12-51.png" width = "480" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图4 示例图像显示了不同患者的检测结果从测试集。检测到的边界框以绿色显示，标准的框以黄色显示。原点位于每个框中心的线段定义相应的坐标系 </font></center>

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211757.jpg" width = "480" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图5. 增加网络深度对病变的分割性能的影响。真阳性，假阴性和假阳性体素分别以绿色，黄色和红色突出显示。由于感受野的大小增加，具有和不具有捷径的7层CEN能够比3层CEN更好地分割大的病变。</font></center>

2018年德国医疗康复机构提出一种具有代表性的基于全卷积的前列腺图像分割方法。用CNN在前列腺的MRI图像上进行端到端训练，并可以一次完成整个分割。提出了一种新的目标函数，在训练期间根据Dice系数进行优化[15]。通过这种方式，可以处理前景和背景之间存在不平衡的情况，并且增加了随机应用的数据非线性变换和直方图匹配。实验评估中表明，该方法在公开数据集上取得了优秀的结果，但大大降低了处理时间。
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211804.jpg" width = "480" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图6 网络架构的示意图</font></center>

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211814.jpg" width = "480" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图7 PROMISE 2012数据集分割结果。</font></center>

3. 图像配准

图象配准是图象融合的前提，是公认难度较大的图象处理技术，也是决定医学图象融合技术发展的关键技术。在临床诊断中，单一模态的图像往往不能提供医生所需要的足够信息，常需将多种模式或同一模式的多次成像通过配准融合来实现感兴趣区的信息互补。在一幅图像上同时表达来自多种成像源的信息，医生就能做出更加准确的诊断或制定出更加合适的治疗方法[16]。医学图像配准包括图像的定位和转换，即通过寻找一种空间变换使两幅图像对应点达到空间位置和解剖结构上的完全一致。图6简单说明了二维图像配准的概念。图(a)和图(b)是对应于同一人脑同一位置的两幅 MRI 图像，其中图(a)是质子密度加权成像，图(b)是纵向弛豫加权成像。这两幅图像有明显的不同，第一是方位上的差异，即图(a)相对于图(b)沿水平和垂直方向分别进行了平移；第二是两幅图像所表达的内容是不一致的，图(a)表达不同组织质子含量的差别，而图(b)则突出不同组织纵向弛豫的差别。图(c)给出了两幅图像之间像素点的对应映射关系，即(a)中的每一个点fx都被映射到(b)中唯一的一个点rx。如果这种映射是一一对应的，即一幅图像空间中的每一个点在另外一幅图像空间中都有对应点，或者至少在医疗诊断上感兴趣的那些点能够准确或近似准确的对应起来，我们就称之为配准[17,18]。图(d)给出了图(a)相对于图(b)的配准图像。从图(d)中可以看出，图(d)与(b)之间的的像素点的空间位置已经近似一致了。1993 年 Petra 等综述了二维图像的配准方法，并根据配准基准的特性,将图像配准的方法分为基于外部特征的图象配准(有框架) 和基于图象内部特征的图象配准(无框架) 两种方法。 后者由于其无创性和可回溯性, 已成为配准算法的研究中心。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211822.jpg" width = "480" height = "150" /></div>
<center><font face="黑体" color=gray size=1>图8 医学图像配准原理 </font></center>

2019年华中科技大学对基于PCANet的结构非刚性多模医学图像配准展开研究。提出了一种基于PCANet的结构表示方法用于多模态医学图像配准[19]。与人工设计的特征提取方法相比，PCANet可以通过多级线性和非线性变换自动从大量医学图像中学习内在特征。所提出的方法可以通过利用PCANet的各个层中提取的多级图像特征来为多模态图像提供有效的结构表示。对Atlas，BrainWeb和RIRE数据集的大量实验表明，与MIND，ESSD，WLD和NMI方法相比，所提出的方法可以提供更低的TRE值和更令人满意的结果

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211830.jpg" width = "480" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图9 第一行分别是x和y方向变形的真实结果，第二行是PSR与x和y方向的真实情况的差异；第三行是MIND方法的变形和真实值之间的差异</font></center>

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211836.jpg" width = "480" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图10 PSR，MIND，ESSD，WLD和NMI方法的CT-MR图像配准。（a）参考PD图像；（b）浮动CT图像；（c）PSR方法；（d）MIND方法；（e）ESSD方法；（f）WLD方法；（g）NMI方法</font></center>

近年来，医学图像配准技术有了新的进展,在配准方法上应用了信息学的理论和方法，例如应用最大化的互信息量作为配准准则进行图像的配准，基于互信息的弹性形变模型也逐渐成为研究热点[20]。在配准对象方面从二维图像发展到三维多模医学图像的配准。一些新算法，如基于小波变换的算法、统计学参数绘图算法、遗传算法等，在医学图像上的应用也在不断扩展。向快速和准确方面改进算法，使用最优化策略改进图像配准以及对非刚性图像配准的研究是今后医学图像配准技 术的发展方向[21,22]。

4. 图像融合

图像融合的主要目的是通过对多幅图像间的冗余数据的处理来提高图像的可读性，对多幅图像间的互补信息的处理来提高图像的清晰度。多模态医学图像的融合把有价值的生理功能信息与精确的解剖结构结合在一起，可以为临床提供更加全面和准确的资料[23]。融合图像的创建分为图像数据的融合与融合图像的显示两部分来完成。目前，图像数据融合主要有以像素为基础的方法和以图像特征为基础的方法。前者是对图像进行逐点处理，把两幅图像对应像素点的灰度值进行加权求和、灰度取大或者灰度取小等操作，算法实现比较简单，不过实现效果和效率都相对较差，融合后图像会出现一定程度的模糊。后者要对图像进行特征提取、目标分割等处理，用到的算法原理复杂，但是实现效果却比较理想。融合图像的显示常用的有伪彩色显示法、断层显示法和三维显示法等。伪彩色显示一般以某个图像为基准,用灰度色阶显示，另一幅图像叠加在基准图像上，用彩色色阶显示。断层显示法常用于某些特定图像，可以将融合后的三维数据以横断面、冠状面和矢状面断层图像同步地显示，便于观察者进行诊断。三维显示法是将融合后数据以三维图像的形式显示，使观察者可更直观地观察病灶的空间解剖位置，这在外科手术设计和放疗计划制定中有重要意义。
<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211845.jpg" width = "200" height = "300" /></div>
<center><font face="黑体" color=gray size=1>图11 医学图像融合阶段的总结。 两阶段过程包括图像配准，然后是图像融合。</font></center>

在图像融合技术研究中，不断有新的方法出现，其中小波变换、 基于有限元分析的非线性配准以及人工智能技术在图像融合中的应用将是今后图像融合研究的热点与方向。随着三维重建显示技术的发展，三维图像融合技术的研究也越来越受到重视，三维图像的融合和信息表达，也将是图像融合研究的一个重点。

在计算机辅助图像处理的基础上，开发出综合利用图像处理方法， 结合人体常数和部分疾病的影像特征来帮助或模拟医生分析、诊断的图像分析系统成为一种必然趋势。目前已有一些采用人机交互定点、自动测量分析的图像分析软件，能定点或定项地完成一些测量和辅助诊断的工作，但远远没有达到智能分析和专家系统的水平；全自动识别标志点并测量分析以及医学图像信息与文本信息的融合， 是计算机辅助诊断技术今后的发展方向。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/20190704211851.jpg" width = "300" height = "380" /></div>
<center><font face="黑体" color=gray size=1>图12 多模态医学图像融合的例子。使用特定图像融合技术的模态1与模态2的组合可以使医学诊断和评估改进</font></center>

5. 预测与挑战

1）数据维度问题-2D与3D：在迄今为止的大多数工作中，是在2D图像中进行处理分析。人们常常质疑向3D过渡是否是迈向性能提高的重要一步。数据增强过程中存在若干变体，包括2.5D。例如，在Roth等人的研究中，以结肠息肉或淋巴结候选体中的体素为中心截取轴向图像，存在冠状和矢状图像。

2）学习方法 - 无监督与监督：当我们查看网络文献时，很明显大多数工作都集中在受监督的CNN上，以实现分类。这种网络对于许多应用是重要的，包括检测，分割和标记。尽管如此，一些工作仍集中于无监督方案，这些方案主要表现为图像编码。诸如玻尔兹曼机器（RBM）之类的无监督表示学习方法可能胜过滤波器，因为它们直接从训练数据中学习特征描述。RBM通过生成学习目标进行培训；这使网络成为可能从未标记的数据中学习，但不一定产生最适合分类的特征。Van Tulder等人进行了一项调查，结合卷积分类和RBM的生成和判别学习目标的优点，该机器学习了对描述训练数据和分类都很好的过滤器。结果表明，学习目标的组合完全胜过生成性学习。

3) 迁移学习和微调：在医学成像领域中获取与ImageNet一样全面注释的数据集仍然是一个挑战。当没有足够的数据时，有几种方法可以继续：1）迁移学习：从自然图像数据集或不同医学领域预训练的CNN模型（监督）用于新的医疗任务。在一个方案中，预先训练CNN应用于输入图像，然后从网络层提取输出。提取的输出被认为是特征并且用于训练单独的模式分类器。2）微调：当手头的任务确实存在中等大小的数据集时，较好的方案是使用预先训练的CNN作为网络的初始化，然后进行进一步的监督训练，其中几个（或全部）网络层，使用任务的新数据。

4）数据隐私受社会和技术问题的影响，需要从社会学和技术学的角度共同解决。在卫生部门讨论隐私时，会想到HIPAA（1996年健康保险流通与责任法案）。它为患者提供有关保护个人身份信息的法律权利，并为医疗保健提供者承担保护和限制其使用或披露的义务。在医疗保健数据不断增加的同时，研究人员面临如何加密患者信息以防止其被使用或披露的问题。同时带来，限制访问数据可能遗漏非常重要的信息。

6、结论
近几年来，与传统的机器学习算法相比，深度学习在日常生活自动化方面占据了中心位置，并取得了相当大的进步。基于优秀的性能，大多数研究人员认为在未来15年内，基于深度学习的应用程序将接管人类和大多数日常活动。但是，与其它现实世界的问题相比，医疗保健领域的深度学习尤其是医学图像的发展速度非常慢。到目前为止深度学习应用提供了积极的反馈，然而，由于医疗保健数据的敏感性和挑战，我们应该寻找更复杂的深度学习方法，以便有效地处理复杂的医疗数据。随着医疗技术和计算机科学的蓬勃发展，对医学图象处理提出的要求也越来越高。有效地提高医学图象处理技术的水平，与多学科理论的交叉融合，医务人员和理论技术人员之间的交流就显得越来越重要。医学图象处理技术作为提升现代医疗诊断水平的有力依据, 使实施风险低、创伤性小的手术方案成为可能，必将在医学信息研究领域发挥更大的作用。

参考文献
[1]林晓, 邱晓嘉. 图像分析技术在医学上的应用 [J] . 包头医学院学报, 2005, 21 (3) ： 311~ 314

[2]周贤善. 医学图像处理技术综述[J]. 福建电脑, 2009(1):34-34.

[3]Mcinerney T , Terzopoulos D . Deformable models in medical image analysis: a survey[J]. Medical Image Analysis, 1996, 1(2):91.

[4]Litjens G , Kooi T , Bejnordi B E , et al. A survey on deep learning in medical image analysis[J]. Medical Image Analysis, 2017, 42:60-88.

[5]Deserno T M , Heinz H , Maier-Hein K H , et al. Viewpoints on Medical Image Processing: From Science to Application[J]. Current Medical Imaging Reviews, 2013, 9(2):79-88.

[6]A. Setio et al., “Pulmonary nodule detection in CT images using multiview convolutional networks,” IEEE Trans. Med. Imag., vol. 35, no. 5,pp. 1160–1169, May 2016.

[7]H. Roth et al., “Improving computer-aided detection using convolutional neural networks and random view aggregation,” IEEE Trans.Med. Imag., vol. 35, no. 5, pp. 1170–1181, May 2016

[8]林瑶, 田捷. 医学图像分割方法综述[J]. 模式识别与人工智能, 2002, 15(2).

[9]Ghesu F C , Georgescu B , Mansi T , et al. An Artificial Agent for Anatomical Landmark Detection in Medical Images[C]// International Conference on Medical Image Computing & Computer-assisted Intervention. Springer, Cham, 2016.

[10]Pham D L , Xu C , Prince J L . Current methods in medical image segmentation.[J]. Annual Review of Biomedical Engineering, 2000, 2(2):315-337.

[11]Lehmann T M , Gonner C , Spitzer K . Survey: interpolation methods in medical image processing[J]. IEEE Transactions on Medical Imaging, 1999, 18(11):1049-1075.

[12]Cootes T F , Taylor C J . Statistical Models of Appearance for Medical Image Analysis and Computer Vision[J]. Proceedings of SPIE - The International Society for Optical Engineering, 2001, 4322(1).

[13] T. Brosch et al., “Deep 3D convolutional encoder networks with shortcuts for multiscale feature integration applied to multiple sclerosis lesion segmentation,” IEEE Trans. Med. Imag., vol. 35, no. 5,pp. 1229–1239, May 2016.

[14]Ghesu F C , Krubasik E , Georgescu B , et al. Marginal Space Deep Learning: Efficient Architecture for Volumetric Image Parsing[J]. IEEE Transactions on Medical Imaging, 2016, 35(5):1217-1228.

[15]Milletari F , Navab N , Ahmadi S A . V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation[J]. 2016.

[16] .周永新, 罗述谦. 一种人机交互式快速脑图象配准系统[J] . 北京生物医学工程, 2002; 21 (1) ：11~14

[17]杨虎, 马斌荣, 任海萍. 基于互信息的人脑图象配准研究[J] . 中国医学物理学杂志, 2001; 18 (2) ：69~73

[18]汪家旺，愈同福，姜晓彤，等.肺部孤立性结节定量研究[J].中国医学影 像技术,2003,19(9)：1218~1219

[19]Ishihara S , Ishihara K , Nagamachi M , et al. An analysis of Kansei structure on shoes using self-organizing neural networks[J]. International Journal of Industrial Ergonomics, 1997, 19(2):93-104.

[20]Maintz J B , Viergever M A . A Survey of Medical Image Registration[J]. Computer & Digital Engineering, 2009, 33(1):140-144.

[21]Hill D L G , Batchelor P G , Holden M , et al. Medical image registration[J]. Physics in Medicine & Biology, 2008, 31(4):1-45.

[22]Razzak M I , Naz S , Zaib A . Deep Learning for Medical Image Processing: Overview, Challenges and Future[J]. 2017.

[23]林晓, 邱晓嘉. 图像分析技术在医学上的应用 [J] . 包头医学院学报, 2005, 21 (3) ： 311~ 314



***

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
***
# 面试经验及技巧
## 数据结构
### 数组
* 数组是一种基本的数据结构，用于**按顺序**存储元素集合。但元素可以随机存取，因为数组中的每个元素都可以通过数组*索引*来识别。
* **动态数组**它仍然是一个随机存取的列表数据结构，但大小是可变的。例如，在 C++ 中的 vector，以及在 Java 中的 ArrayList。

### Hash table(散列表)

## 算法

### 快速排序算法
快速排序是由东尼·霍尔所发展的一种排序算法。在平均状况下，排序 n 个项目要Ο(n log n)次比较。在最坏状况下则需要Ο(_n_2)次比较，但这种状况并不常见。事实上，快速排序通常明显比其他Ο(n log n) 算法更快，因为它的内部循环（inner loop）可以在大部分的架构上很有效率地被实现出来。
快速排序使用分治法（Divide and conquer）策略来把一个串行（list）分为两个子串行（sub-lists）。\
**算法步骤：**
1 从数列中挑出一个元素，称为 “基准”（pivot），
2 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。
3 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。
递归的最底部情形，是数列的大小是零或一，也就是永远都已经被排序好了。虽然一直递归下去，但是这个算法总会退出，因为在每次的迭代（iteration）中，它至少会把一个元素摆到它最后的位置去。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/sort_quick_anim.gif" width = "480" height = "300" /></div>


### 堆排序算法
堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：**即子结点的键值或索引总是小于（或者大于）它的父节点。**
堆排序的平均时间复杂度为Ο(n_log_n) 
**算法步骤:**
1 创建一个堆H[0...n-1]
2 把堆首（最大值）和堆尾互换
3 把堆的尺寸缩小1，并调用shift_down(0),目的是把新的数组顶端数据调整到相应位置
4 重复步骤2，直到堆的尺寸为1

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/heap_sort.gif" width = "480" height = "300" /></div>

### 归并排序
归并排序（Merge sort，台湾译作：合并排序）是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。
**算法步骤：**
1. 申请空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列
2. 设定两个指针，最初位置分别为两个已经排序序列的起始位置
3. 比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置
4. 重复步骤3直到某一指针达到序列尾
5. 将另一序列剩下的所有元素直接复制到合并序列尾

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/merge_sort.gif" width = "480" height = "300" /></div>

### 二分查找算法
二分查找算法是一种在有序数组中查找某一特定元素的搜索算法。搜素过程从数组的中间元素开始，如果中间元素正好是要查找的元素，则搜素过程结束；如果某一特定元素大于或者小于中间元素，则在数组大于或小于中间元素的那一半中查找，而且跟开始一样从中间元素开始比较。如果在某一步骤数组为空，则代表找不到。这种搜索算法每一次比较都使搜索范围缩小一半。折半搜索每次把搜索区域减少一半，时间复杂度为Ο(log_n_) 。

### 线性查找算法（BFPRT）
BFPRT算法解决的问题十分经典，即从某n个元素的序列中选出第k大（第k小）的元素，通过巧妙的分析，BFPRT可以保证在最坏情况下仍为线性时间复杂度。该算法的思想与快速排序思想相似，当然，为使得算法在最坏情况下，依然能达到o(n)的时间复杂度，五位算法作者做了精妙的处理。
**算法步骤：**

1. 将n个元素每5个一组，分成n/5(上界)组。
2. 取出每一组的中位数，任意排序方法，比如插入排序。
3. 递归的调用selection算法查找上一步中所有中位数的中位数，设为x，偶数个中位数的情况下设定为选取中间小的一个。
4. 用x来分割数组，设小于等于x的个数为k，大于x的个数即为n-k。
5. 若i==k，返回x；若i<k，在小于x的元素中递归查找第i小的元素；若i>k，在大于x的元素中递归查找第i-k小的元素。

终止条件：n=1时，返回的即是i小元素。


### 图遍历
* 深度优先搜索（Depth-First-Search，DFS）
深度优先搜索（缩写DFS）有点类似广度优先搜索，也是对一个连通图进行遍历的算法。它的思想是从一个顶点V0开始，沿着一条路一直走到底，如果发现不能到达目标解，那就返回到上一个节点，然后从另一条路开始走到底，这种尽量往深处走的概念即是深度优先的概念。
是搜索算法的一种。它沿着树的深度遍历树的节点，尽可能深的搜索树的分支。当节点v的所有边都己被探寻过，搜索将回溯到发现节点v的那条边的起始节点。这一过程一直进行到已发现从源节点可达的所有节点为止。如果还存在未被发现的节点，则选择其中一个作为源节点并重复以上过程，整个进程反复进行直到所有节点都被访问为止。DFS属于盲目搜索。
深度优先搜索是图论中的经典算法，利用深度优先搜索算法可以产生目标图的相应拓扑排序表，利用拓扑排序表可以方便的解决很多相关的图论问题，如最大路径问题等等。一般用堆数据结构来辅助实现DFS算法。
深度优先遍历图算法步骤：

1. 访问顶点v；
2. 依次从v的未被访问的邻接点出发，对图进行深度优先遍历；直至图中和v有路径相通的顶点都被访问；
3. 若此时图中尚有顶点未被访问，则从一个未被访问的顶点出发，重新进行深度优先遍历，直到图中所有顶点均被访问过为止。

上述描述可能比较抽象，举个实例：
DFS 在访问图中某一起始顶点 v 后，由 v 出发，访问它的任一邻接顶点 w1；再从 w1 出发，访问与 w1邻 接但还没有访问过的顶点 w2；然后再从 w2 出发，进行类似的访问，… 如此进行下去，直至到达所有的邻接顶点都被访问过的顶点 u 为止。
接着，退回一步，退到前一次刚访问过的顶点，看是否还有其它没有被访问的邻接顶点。如果有，则访问此顶点，之后再从此顶点出发，进行与前述类似的访问；如果没有，就再退回一步进行搜索。重复上述过程，直到连通图中所有顶点都被访问过为止。

* 广度优先搜索（Breadth-first search，BFS）
广度优先搜索一个图的时候是按照树的层次来搜索的，（层次遍历），队列来实现，形象的说，这里有点像辐射形状的搜索方式，从一个节点，向其旁边节点传递病毒，就这样一层一层的传递辐射下去，知道目标节点被辐射中了，此时就已经找到了从起点到终点的路径。
简单的说，BFS是从根节点开始，沿着树(图)的宽度遍历树(图)的节点。如果所有节点均被访问，则算法中止。BFS同样属于盲目搜索。一般用队列数据结构来辅助实现BFS算法。

**算法步骤：**
1. 首先将根节点放入队列中。
2. 从队列中取出第一个节点，并检验它是否为目标。
    * 如果找到目标，则结束搜寻并回传结果。
    * 否则将它所有尚未检验过的直接子节点加入队列中。
3. 若队列为空，表示整张图都检查过了——亦即图中没有欲搜寻的目标。结束搜寻并回传“找不到目标”。
4. 重复步骤2。

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/BFS.gif" width = "480" height = "380" /></div>


### Dijkstra算法
戴克斯特拉算法（Dijkstra’s algorithm）是由荷兰计算机科学家艾兹赫尔·戴克斯特拉提出。迪科斯彻算法使用了广度优先搜索解决非负权有向图的单源最短路径问题，算法最终得到一个最短路径树。该算法常用于路由算法或者作为其他图算法的一个子模块。
该算法的输入包含了一个有权重的有向图 G，以及G中的一个来源顶点 S。我们以 V 表示 G 中所有顶点的集合。每一个图中的边，都是两个顶点所形成的有序元素对。(u, v) 表示从顶点 u 到 v 有路径相连。我们以 E 表示G中所有边的集合，而边的权重则由权重函数 w: E → [0, ∞] 定义。因此，w(u, v) 就是从顶点 u 到顶点 v 的非负权重（weight）。边的权重可以想像成两个顶点之间的距离。任两点间路径的权重，就是该路径上所有边的权重总和。已知有 V 中有顶点 s 及 t，Dijkstra 算法可以找到 s 到 t的最低权重路径(例如，最短路径)。这个算法也可以在一个图中，找到从一个顶点 s 到任何其他顶点的最短路径。对于不含负权的有向图，Dijkstra算法是目前已知的最快的单源最短路径算法。
算法步骤：

初始时令 S={V0},T={其余顶点}，T中顶点对应的距离值

若存在<V0,Vi>，d(V0,Vi)为<V0,Vi>弧上的权值
若不存在<V0,Vi>，d(V0,Vi)为∞

从T中选取一个其距离值为最小的顶点W且不在S中，加入S
对其余T中顶点的距离值进行修改：若加进W作中间顶点，从V0到Vi的距离值缩短，则修改此距离值

重复上述步骤2、3，直到S中包含所有顶点，即W=Vi为止

<div align=center><img src="https://raw.githubusercontent.com/axjing/axjingWorks/master/Reference/Dijkstra.gif" width = "480" height = "380" /></div>

### 动态规划算法
动态规划（Dynamic programming）是一种在数学、计算机科学和经济学中使用的，通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。 动态规划常常适用于有重叠子问题和最优子结构性质的问题，动态规划方法所耗时间往往远少于朴素解法。
动态规划背后的基本思想非常简单。大致上，若要解一个给定问题，我们需要解其不同部分（即子问题），再合并子问题的解以得出原问题的解。 通常许多子问题非常相似，为此动态规划法试图仅仅解决每个子问题一次，从而减少计算量： 一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。 这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。
关于动态规划最经典的问题当属背包问题。
**算法步骤：**

1 最优子结构性质。如果问题的最优解所包含的子问题的解也是最优的，我们就称该问题具有最优子结构性质（即满足最优化原理）。最优子结构性质为动态规划算法解决问题提供了重要线索。
2 子问题重叠性质。子问题重叠性质是指在用递归算法自顶向下对问题进行求解时，每次产生的子问题并不总是新问题，有些子问题会被重复计算多次。动态规划算法正是利用了这种子问题的重叠性质，对每一个子问题只计算一次，然后将其计算结果保存在一个表格中，当再次需要计算已经计算过的子问题时，只是在表格中简单地查看一下结果，从而获得较高的效率。

### 朴素贝叶斯分类算法
朴素贝叶斯分类算法是一种基于贝叶斯定理的简单概率分类算法。贝叶斯分类的基础是概率推理，就是在各种条件的存在不确定，仅知其出现概率的情况下，如何完成推理和决策任务。概率推理是与确定性推理相对应的。而朴素贝叶斯分类器是基于独立假设的，即假设样本每个特征与其他特征都不相关。
朴素贝叶斯分类器依靠精确的自然概率模型，在有监督学习的样本集中能获取得非常好的分类效果。在许多实际应用中，朴素贝叶斯模型参数估计使用最大似然估计方法，换言之朴素贝叶斯模型能工作并没有用到贝叶斯概率或者任何贝叶斯模型。
尽管是带着这些朴素思想和过于简单化的假设，但朴素贝叶斯分类器在很多复杂的现实情形中仍能够取得相当好的效果。