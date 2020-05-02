"""
1.选中Tensorboard左上角的 Show data download links
2.选中右下角的下载文件的格式，选择的csv格式
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import interpolate
from matplotlib.font_manager import FontProperties
import csv
from decimal import Decimal
#mpl.use("TKAgg")
#mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
mpl.use("TKAgg")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
myfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串
#myfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')
'''读取csv文件'''
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots: 
        #print(y)
        if row[1] == "Step":
            print(row)
        elif int(str(row[1]))<40000:
            x.append(int(str(row[1])))
            y.append(float(row[2]))
        elif int(str(row[1]))>40000 and int(str(row[1]))<120000:
            if float(row[2])<0.4:

        
                x.append(int(str(row[1])))
                y.append(float(row[2]))
            #print(row[1], row[2])
    return x[::2] ,y[::2]

def SmoothLine(x_arry, y_arry, scaling, option=1):
    x_smooth = np.linspace(0, max(x_arry), max(x_arry) * scaling)
    if option == 0:
        y_smooth = y_arry
    elif option==1:
        spl = interpolate.splrep(x_arry, y_arry)
        y_smooth = interpolate.splev(x_smooth,spl)
    elif option==2:
        y_smooth = interpolate.spline(x_arry, y_arry, x_smooth)
    return x_smooth, y_smooth
 
if __name__ == "__main__":
    
    #LearningRate_IRNet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-LearningRate_LearningRate_learning_rate.csv"
    #TotalLoss_IRNet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-Losses_TotalLoss.csv"
    workspace="/home/andy/anaconda3/ANCODE/axjingWorks/workspace/AcademicAN/paper_img/Loss/"
    IRNet_BoxLoss_Classification_path = workspace+"T/inceptionResnet/run_.-tag-Losses_Loss_BoxClassifierLoss_classification_loss.csv"
    IRNet_BoxLoss_Localization_path = workspace+"T/inceptionResnet/run_.-tag-Losses_Loss_BoxClassifierLoss_localization_loss.csv"
    IRNet_RPNLoss_Localization_path = workspace+"T/inceptionResnet/run_.-tag-Losses_Loss_RPNLoss_localization_loss.csv"
    IRNet_RPNLoss_Objectness_path = workspace+"T/inceptionResnet/run_.-tag-Losses_Loss_RPNLoss_objectness_loss.csv"

    inc_BoxLoss_Classification_path = workspace+"T/Inception/run_.-tag-Losses_Loss_BoxClassifierLoss_classification_loss.csv"
    inc_BoxLoss_Localization_path = workspace+"T/Inception/run_.-tag-Losses_Loss_BoxClassifierLoss_localization_loss.csv"
    inc_RPNLoss_Localization_path = workspace+"T/Inception/run_.-tag-Losses_Loss_RPNLoss_localization_loss.csv"
    inc_RPNLoss_Objectness_path = workspace+"T/Inception/run_.-tag-Losses_Loss_RPNLoss_objectness_loss.csv"

    R_BoxLoss_Classification_path = workspace+"T/ResNet/run_.-tag-Losses_Loss_BoxClassifierLoss_classification_loss.csv"
    R_BoxLoss_Localization_path = workspace+"T/ResNet/run_.-tag-Losses_Loss_BoxClassifierLoss_localization_loss.csv"
    R_RPNLoss_Localization_path = workspace+"T/ResNet/run_.-tag-Losses_Loss_RPNLoss_localization_loss.csv"
    R_RPNLoss_Objectness_path = workspace+"T/ResNet/run_.-tag-Losses_Loss_RPNLoss_objectness_loss.csv"

    
    #f, ax = plt.subplots(3,2)
    #style_list = ["g+-", "r*-", "b.-", "yo-", "s--"]
    style_list = ["g-", "c:", "y-.", "y-", "s--"]
    #ax1 = figure.add_subplot(221)

    
    x_BoxLoss_Classification_IRNet, y_BoxLoss_Classification_IRNet = readcsv(IRNet_BoxLoss_Classification_path)
    x_BoxLoss_Classification_IRNet_smooths, y_BoxLoss_Classification_IRNet_smooths = SmoothLine(x_BoxLoss_Classification_IRNet, y_BoxLoss_Classification_IRNet, scaling=10, option=0)
    
    x_BoxLoss_Classification_inc, y_BoxLoss_Classification_inc = readcsv(inc_BoxLoss_Classification_path)
    x_BoxLoss_Classification_inc_smooths, y_BoxLoss_Classification_inc_smooths = SmoothLine(x_BoxLoss_Classification_inc, y_BoxLoss_Classification_inc, scaling=10, option=0)
    
    x_BoxLoss_Classification_R, y_BoxLoss_Classification_R = readcsv(R_BoxLoss_Classification_path)
    x_BoxLoss_Classification_inc_smooths, y_BoxLoss_Classification_inc_smooths = SmoothLine(x_BoxLoss_Classification_R, y_BoxLoss_Classification_R, scaling=10, option=0)

    figure = plt.figure(figsize=(12,7))
    plt.suptitle("Transverse US Train Loss", fontsize=12)
    plt.subplot(221)
    plt.title("Box Classification Loss")
    plt.plot(x_BoxLoss_Classification_IRNet, y_BoxLoss_Classification_IRNet, style_list[0], label="incepResNet")
    plt.plot(x_BoxLoss_Classification_inc, y_BoxLoss_Classification_inc, style_list[1], label="Inception")
    plt.plot(x_BoxLoss_Classification_R, y_BoxLoss_Classification_R, style_list[2], label="ResNet")
    plt.legend(fontsize=12)
    # plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    #plt.plot(x_BoxLoss_Classification_IRNet, y_BoxLoss_Localization_IRNet, '-', color='g', label='BoxLossClassification')
    
    
    x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet = readcsv(IRNet_BoxLoss_Localization_path)
    x_BoxLoss_Localization_IRNet_smooths, y_BoxLoss_Localization_IRNet_smooths = SmoothLine(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, scaling=10, option=1)

    x_BoxLoss_Localization_inc, y_BoxLoss_Localization_inc = readcsv(inc_BoxLoss_Localization_path)
    x_BoxLoss_Localization_inc_smooths, y_BoxLoss_Localization_inc_smooths = SmoothLine(x_BoxLoss_Localization_inc, y_BoxLoss_Localization_inc, scaling=10, option=1)

    x_BoxLoss_Localization_R, y_BoxLoss_Localization_R = readcsv(R_BoxLoss_Localization_path)
    x_BoxLoss_Localization_R_smooths, y_BoxLoss_Localization_R_smooths = SmoothLine(x_BoxLoss_Localization_R, y_BoxLoss_Localization_R, scaling=10, option=1)

    plt.subplot(222)
    plt.title("Box Regression Loss")
    plt.plot(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, style_list[0], label='incepResNet')
    plt.plot(x_BoxLoss_Localization_inc, y_BoxLoss_Localization_inc, style_list[1], label='Inception')    
    plt.plot(x_BoxLoss_Localization_R, y_BoxLoss_Localization_R, style_list[2], label='ResNet')    
    plt.legend(fontsize=12)
    # plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    #plt.plot(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, '--', color='black', label='BoxLossLocalization')

    
    x_RPNLoss_Objectness_IRnet, y_RPNLoss_Objectness_IRNet = readcsv(IRNet_RPNLoss_Objectness_path)
    x_RPNLoss_Objectness_IRnet_smooths, y_RPNLoss_Objectness_IRNet_smooths = SmoothLine(x_RPNLoss_Objectness_IRnet, y_RPNLoss_Objectness_IRNet, scaling=10, option=1)
    
    x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc = readcsv(inc_RPNLoss_Objectness_path)
    x_RPNLoss_Objectness_R_smooths, y_RPNLoss_Objectness_R_smooths = SmoothLine(x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc, scaling=0.1, option=1)

    x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R = readcsv(R_RPNLoss_Localization_path)
    x_RPNLoss_Objectness_R_smooths, y_RPNLoss_Objectness_R_smooths = SmoothLine(x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R, scaling=0.1, option=1)


    plt.subplot(223)
    plt.title("RPN Classification Loss")
    plt.plot(x_RPNLoss_Objectness_IRnet, y_RPNLoss_Objectness_IRNet, style_list[0], label='incepResNet')
    plt.plot(x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc, style_list[1], label='Inception')
    plt.plot(x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R, style_list[2], label='ResNet')
    plt.legend(fontsize=12)
    plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    #plt.plot(x_RPNLoss_Localization_IRnet_smooths, y_RPNLoss_Localization_IRNet_smooths , color='blue', label='RPNLossLocalization')

    
    x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet = readcsv(IRNet_RPNLoss_Objectness_path)
    x_RPNLoss_Objectness_IRNet_smooths, y_RPNLoss_Objectness_IRNet_smooths = SmoothLine(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, scaling=10, option=1)
    
    x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc = readcsv(inc_RPNLoss_Objectness_path)
    x_RPNLoss_Objectness_IRNet_smooths, y_RPNLoss_Objectness_IRNet_smooths = SmoothLine(x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc, scaling=10, option=1)

    x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R = readcsv(R_RPNLoss_Objectness_path)
    x_RPNLoss_Objectness_R_smooths, y_RPNLoss_Objectness_R_smooths = SmoothLine(x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R, scaling=10, option=1)

    plt.subplot(224)
    #plt.plot(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, style_list[3], label='RPNLoss_Objectness')
    plt.title("RPNLoss_Objectness")
    plt.plot(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, style_list[0], label='incepResNet')
    plt.plot(x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc, style_list[1], label='Inception')
    plt.plot(x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R, style_list[2], label='ResNet')
    plt.legend(fontsize=12)
    plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    
    plt.savefig(workspace+"T/横向.svg", format="svg")
    # plt.show()


    #****************************#
    #*************纵向***********#
    #****************************#
    IRNet_BoxLoss_Classification_path_L = workspace+"L/inceptionResnet/run_.-tag-Losses_Loss_BoxClassifierLoss_classification_loss.csv"
    IRNet_BoxLoss_Localization_path_L = workspace+"L/inceptionResnet/run_.-tag-Losses_Loss_BoxClassifierLoss_localization_loss.csv"
    IRNet_RPNLoss_Localization_path_L = workspace+"L/inceptionResnet/run_.-tag-Losses_Loss_RPNLoss_localization_loss.csv"
    IRNet_RPNLoss_Objectness_path_L = workspace+"L/inceptionResnet/run_.-tag-Losses_Loss_RPNLoss_objectness_loss.csv"

    inc_BoxLoss_Classification_path_L = workspace+"L/Inception/run_.-tag-Losses_Loss_BoxClassifierLoss_classification_loss.csv"
    inc_BoxLoss_Localization_path_L = workspace+"L/Inception/run_.-tag-Losses_Loss_BoxClassifierLoss_localization_loss.csv"
    inc_RPNLoss_Localization_path_L = workspace+"L/Inception/run_.-tag-Losses_Loss_RPNLoss_localization_loss.csv"
    inc_RPNLoss_Objectness_path_L = workspace+"L/Inception/run_.-tag-Losses_Loss_RPNLoss_objectness_loss.csv"

    R_BoxLoss_Classification_path_L = workspace+"L/ResNet/run_.-tag-Losses_Loss_BoxClassifierLoss_classification_loss.csv"
    R_BoxLoss_Localization_path_L = workspace+"L/ResNet/run_.-tag-Losses_Loss_BoxClassifierLoss_localization_loss.csv"
    R_RPNLoss_Localization_path_L = workspace+"L/ResNet/run_.-tag-Losses_Loss_RPNLoss_localization_loss.csv"
    R_RPNLoss_Objectness_path_L = workspace+"L/ResNet/run_.-tag-Losses_Loss_RPNLoss_objectness_loss.csv"

    
    x_BoxLoss_Classification_IRNet_L, y_BoxLoss_Classification_IRNet_L = readcsv(IRNet_BoxLoss_Classification_path_L)
    #x_BoxLoss_Classification_IRNet_smooths_L, y_BoxLoss_Classification_IRNet_smooths_L = SmoothLine(x_BoxLoss_Classification_IRNet_L, y_BoxLoss_Classification_IRNet_L, scaling=10, option=0)
    
    x_BoxLoss_Classification_inc_L, y_BoxLoss_Classification_inc_L = readcsv(inc_BoxLoss_Classification_path_L)
    #x_BoxLoss_Classification_inc_smooths, y_BoxLoss_Classification_inc_smooths = SmoothLine(x_BoxLoss_Classification_inc, y_BoxLoss_Classification_inc, scaling=10, option=0)
    
    x_BoxLoss_Classification_R_L, y_BoxLoss_Classification_R_L = readcsv(R_BoxLoss_Classification_path_L)
    #x_BoxLoss_Classification_inc_smooths, y_BoxLoss_Classification_inc_smooths = SmoothLine(x_BoxLoss_Classification_R, y_BoxLoss_Classification_R, scaling=10, option=0)

    figure = plt.figure(figsize=(12,7))
    plt.suptitle("Longitudinal US Train Loss")
    plt.subplot(221)
    plt.title("Box Classification Loss")
    plt.plot(x_BoxLoss_Classification_IRNet_L, y_BoxLoss_Classification_IRNet_L, style_list[0], label="incepResNet")
    plt.plot(x_BoxLoss_Classification_inc_L, y_BoxLoss_Classification_inc_L, style_list[1], label="Inception")
    plt.plot(x_BoxLoss_Classification_R_L, y_BoxLoss_Classification_R_L, style_list[2], label="ResNet")
    plt.legend(fontsize=12)
    #plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    #plt.plot(x_BoxLoss_Classification_IRNet, y_BoxLoss_Localization_IRNet, '-', color='g', label='BoxLossClassification')
    
    
    x_BoxLoss_Localization_IRNet_L, y_BoxLoss_Localization_IRNet_L = readcsv(IRNet_BoxLoss_Localization_path_L)
    #x_BoxLoss_Localization_IRNet_smooths, y_BoxLoss_Localization_IRNet_smooths = SmoothLine(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, scaling=10, option=1)

    x_BoxLoss_Localization_inc_L, y_BoxLoss_Localization_inc_L = readcsv(inc_BoxLoss_Localization_path_L)
    #x_BoxLoss_Localization_inc_smooths, y_BoxLoss_Localization_inc_smooths = SmoothLine(x_BoxLoss_Localization_inc, y_BoxLoss_Localization_inc, scaling=10, option=1)

    x_BoxLoss_Localization_R_L, y_BoxLoss_Localization_R_L = readcsv(R_BoxLoss_Localization_path_L)
    #x_BoxLoss_Localization_R_smooths, y_BoxLoss_Localization_R_smooths = SmoothLine(x_BoxLoss_Localization_R, y_BoxLoss_Localization_R, scaling=10, option=1)

    plt.subplot(222)
    plt.title("Box Regression Loss")
    plt.plot(x_BoxLoss_Localization_IRNet_L, y_BoxLoss_Localization_IRNet_L, style_list[0], label='incepResNet')
    plt.plot(x_BoxLoss_Localization_inc_L, y_BoxLoss_Localization_inc_L, style_list[1], label='Inception')    
    plt.plot(x_BoxLoss_Localization_R_L, y_BoxLoss_Localization_R_L, style_list[2], label='ResNet')    
    plt.legend(fontsize=12)
    #plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    #plt.plot(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, '--', color='black', label='BoxLossLocalization')

    
    x_RPNLoss_Objectness_IRnet_L, y_RPNLoss_Objectness_IRNet_L = readcsv(IRNet_RPNLoss_Objectness_path_L)
    #x_RPNLoss_Objectness_IRnet_smooths, y_RPNLoss_Objectness_IRNet_smooths = SmoothLine(x_RPNLoss_Objectness_IRnet, y_RPNLoss_Objectness_IRNet, scaling=10, option=1)
    
    x_RPNLoss_Objectness_inc_L, y_RPNLoss_Objectness_inc_L = readcsv(inc_RPNLoss_Objectness_path_L)
    #x_RPNLoss_Objectness_R_smooths, y_RPNLoss_Objectness_R_smooths = SmoothLine(x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc, scaling=0.1, option=1)

    x_RPNLoss_Objectness_R_L, y_RPNLoss_Objectness_R_L = readcsv(R_RPNLoss_Localization_path_L)
    #x_RPNLoss_Objectness_R_smooths, y_RPNLoss_Objectness_R_smooths = SmoothLine(x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R, scaling=0.1, option=1)


    plt.subplot(223)
    plt.title("RPN Classification Loss")
    plt.plot(x_RPNLoss_Objectness_IRnet_L, y_RPNLoss_Objectness_IRNet_L, style_list[0], label='incepResNet')
    plt.plot(x_RPNLoss_Objectness_inc_L, y_RPNLoss_Objectness_inc_L, style_list[1], label='Inception')
    plt.plot(x_RPNLoss_Objectness_R_L, y_RPNLoss_Objectness_R_L, style_list[2], label='ResNet')
    plt.legend(fontsize=12)
    plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    #plt.plot(x_RPNLoss_Localization_IRnet_smooths, y_RPNLoss_Localization_IRNet_smooths , color='blue', label='RPNLossLocalization')

    
    x_RPNLoss_Objectness_IRNet_L, y_RPNLoss_Objectness_IRNet_L = readcsv(IRNet_RPNLoss_Objectness_path_L)
    #x_RPNLoss_Objectness_IRNet_smooths, y_RPNLoss_Objectness_IRNet_smooths = SmoothLine(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, scaling=10, option=1)
    
    x_RPNLoss_Objectness_inc_L, y_RPNLoss_Objectness_inc_L = readcsv(inc_RPNLoss_Objectness_path_L)
    #x_RPNLoss_Objectness_IRNet_smooths, y_RPNLoss_Objectness_IRNet_smooths = SmoothLine(x_RPNLoss_Objectness_inc, y_RPNLoss_Objectness_inc, scaling=10, option=1)

    x_RPNLoss_Objectness_R_L, y_RPNLoss_Objectness_R_L = readcsv(R_RPNLoss_Objectness_path_L)
    #x_RPNLoss_Objectness_R_smooths, y_RPNLoss_Objectness_R_smooths = SmoothLine(x_RPNLoss_Objectness_R, y_RPNLoss_Objectness_R, scaling=10, option=1)

    plt.subplot(224)
    #plt.plot(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, style_list[3], label='RPNLoss_Objectness')
    plt.title("RPNLoss_Objectness")
    plt.plot(x_RPNLoss_Objectness_IRNet_L, y_RPNLoss_Objectness_IRNet_L, style_list[0], label='incepResNet')
    plt.plot(x_RPNLoss_Objectness_inc_L, y_RPNLoss_Objectness_inc_L, style_list[1], label='Inception')
    plt.plot(x_RPNLoss_Objectness_R_L, y_RPNLoss_Objectness_R_L, style_list[2], label='ResNet')
    plt.legend(fontsize=12)
    plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.grid(ls='--')
    
    plt.savefig(workspace+"L/纵向.svg", format="svg")
    plt.show()
