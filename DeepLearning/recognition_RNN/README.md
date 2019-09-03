目前为单层LSTM对降采样后的数据进行识别，目前训练集识别率80%左右。

后续可能需要做的：
1.对数据降采样前进行带通滤波处理；
2.增加LSTM层数；
3.稍加大时间步，使之覆盖4秒左右的数据；
4.可改为GRU来降低计算量；
5.用C++实现模型预测；
6.对模型预测的数据进行纠错；

文件说明
globalVar.py:
    文件环境全局变量
data_faction_for_rnn.py:
    数据处理文件，会生成out目录下的训练集数据与测试集数据，需要cut_data与uniformed_data文件夹。需要注意的是在其中有TODO，TODO处需要在降采样前需要先过带通滤波器
RNN_bohai.py:
    训练主文件，需要先运行data_faction_for_rnn.py生成out目录下的训练集数据与测试集数据
plt_datas_label.py:
    看图脚本，可以查看切分与原图是否正确