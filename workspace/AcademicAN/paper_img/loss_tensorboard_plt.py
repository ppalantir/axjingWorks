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
        else:
            x.append(int(str(row[1])))
            y.append(float(row[2]))
            #print(row[1], row[2])
    return x ,y

def SmoothLine(x_arry, y_arry, scaling, option=1):
    x_smooth = np.linspace(0, max(x_arry), max(x_arry) * scaling)
    if option==1:
        spl = interpolate.splrep(x_arry, y_arry)
        y_smooth = interpolate.splev(x_smooth,spl)
    elif option==2:
        y_smooth = interpolate.spline(x_arry, y_arry, x_smooth)
    return x_smooth, y_smooth
 
if __name__ == "__main__":
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    LearningRate_IRNet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-LearningRate_LearningRate_learning_rate.csv"
    TotalLoss_IRNet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-Losses_TotalLoss.csv"
    BoxLoss_Classification_IRNet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-Losses_Loss_BoxClassifierLoss_classification_loss.csv"
    BoxLoss_Localization_IRNet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-Losses_Loss_BoxClassifierLoss_localization_loss.csv"
    RPNLoss_Localization_IRnet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-Losses_Loss_RPNLoss_localization_loss.csv"
    RPNLoss_Objectness_IRNet_path = "/home/andy/anaconda3/envs/tensorflow-gpu/axingWorks/ANcWork/Transverse/inceptionResNetTrain/result/run_.-tag-Losses_Loss_RPNLoss_objectness_loss.csv"

    #figure = plt.figure()
    f, ax = plt.subplots(3,2)
    style_list = ["g+-", "r*-", "b.-", "yo-", "s--"]

    #ax1 = figure.add_subplot(221)

    
    x_BoxLoss_Classification_IRNet, y_BoxLoss_Classification_IRNet = readcsv(BoxLoss_Classification_IRNet_path)
    x_BoxLoss_Classification_IRNet_smooths, y_BoxLoss_Classification_IRNet_smooths = SmoothLine(x_BoxLoss_Classification_IRNet, y_BoxLoss_Classification_IRNet, scaling=10, option=1)
    plt.xticks(fontsize=12, rotation=60)
    plt.yticks(fontsize=12)
    plt.xlim(min(x_BoxLoss_Classification_IRNet), max(x_BoxLoss_Classification_IRNet))
    plt.ylim(min(y_BoxLoss_Classification_IRNet)-0.05, max(y_BoxLoss_Classification_IRNet)-0.005)
    plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    ax[0][0].plot(x_BoxLoss_Classification_IRNet, y_BoxLoss_Classification_IRNet, style_list[0], label="TotalLoss")
    ax[0][0].legend(fontsize=12)
    #plt.plot(x_BoxLoss_Classification_IRNet, y_BoxLoss_Localization_IRNet, '-', color='g', label='BoxLossClassification')
    
    
    x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet = readcsv(BoxLoss_Localization_IRNet_path)
    x_BoxLoss_Localization__smooths, y_BoxLoss_Localization_IRNet_smooths = SmoothLine(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, scaling=10, option=1)
    ax[0][1].plot(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, style_list[1], label='BoxLossLocalization')
    #plt.plot(x_BoxLoss_Localization_IRNet, y_BoxLoss_Localization_IRNet, '--', color='black', label='BoxLossLocalization')

    
    x_RPNLoss_Localization_IRnet, y_RPNLoss_Localization_IRNet = readcsv(RPNLoss_Localization_IRnet_path)
    x_RPNLoss_Localization_IRnet_smooths, y_RPNLoss_Localization_IRNet_smooths = SmoothLine(x_RPNLoss_Localization_IRnet, y_RPNLoss_Localization_IRNet, scaling=10, option=1)
    ax[1][0].plot(x_RPNLoss_Localization_IRnet, y_RPNLoss_Localization_IRNet, style_list[2], label='RPNLossLocalization')
    #plt.plot(x_RPNLoss_Localization_IRnet, y_RPNLoss_Localization_IRnet, color='blue', label='RPNLossLocalization')

    
    x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet = readcsv(RPNLoss_Objectness_IRNet_path)
    x_RPNLoss_Objectness_IRNet_smooths, y_RPNLoss_Objectness_IRNet_smooths = SmoothLine(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, scaling=10, option=1)
    ax[1][1].plot(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, style_list[3], label='RPNLoss_Objectness')
    #plt.plot(x_RPNLoss_Objectness_IRNet, y_RPNLoss_Objectness_IRNet, color='bisque', label='RPNLoss_Objectness')
    
    x_TotalLoss_IRNet, y_TotalLoss_IRNet = readcsv(TotalLoss_IRNet_path)
    x_TotalLoss_IRNet_smooths, y_TotalLoss_IRNet_smooths = SmoothLine(x_TotalLoss_IRNet, y_TotalLoss_IRNet, scaling=10, option=1)
    ax[2][0].plot(x_TotalLoss_IRNet, y_TotalLoss_IRNet, style_list[4], label="TotalLoss")
    #plt.plot(x_TotalLoss_IRNet, y_TotalLoss_IRNet, "-", color='red', label='TotalLoss')

    x_LearningRate_IRNet, y_LearningRate_IRNet = readcsv(LearningRate_IRNet_path)
    x_LearningRate_IRNet_smooths, y_LearningRate_IRNet_smooths = SmoothLine(x_LearningRate_IRNet, y_LearningRate_IRNet, scaling=10, option=1)
    ax[2][1].plot(x_LearningRate_IRNet, y_LearningRate_IRNet, style_list[4], label="TotalLoss")
    #plt.plot(x_TotalLoss_IRNet, y_TotalLoss_IRNet, "-", color='red', label='TotalLoss')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(min(x_TotalLoss_IRNet), max(x_TotalLoss_IRNet))
    plt.ylim(min(y_TotalLoss_IRNet)-0.05, max(y_TotalLoss_IRNet)-0.005)
    
    plt.xlabel('Steps',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.legend(fontsize=12)
    
    plt.show()
