'''
@Author: your name
@Date: 2020-02-28 23:01:39
@LastEditTime: 2020-02-28 23:01:48
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \axjingWorks\algorithm_note\huawei\thoughWorks.py
'''
import numpy as np
shape=[4,4]
detect=[['G','F','R','G','-'],['G','-','R','F','R'],['G','F','R','F','R'],['G','-','-','F','R'],['G','R','-','-','F'],['G','G','R','-','-']]
#RGRG
#FFFF
#GRGR
#GRFG
shape1=[4,5]
detect1=[['G','F','R','F','-'],['G','-','R','F','R'],['G','F','R','R','R'],['G','-','G','F','R'],['F','G','F','R','F'],['R','F','R','F','G'],['F','R','G','-','R'],['F','G','G','-','-']]
#RGRGG
#FFFFF
#GRGRR
#FGRFG
class wurenji():
    def __init__(self):
        self.dire=0
        self.dirlis=[0,1,2]#up,down,left
    def buildmap(self,shape,string):
        n=shape[0]
        m=shape[1]
        numdic={'G':1,'F':2,'R':3}
        strdic={1:'G',2:'F',3:'R'}
        tmp=np.zeros((n,m))
        start=[n-2,0]
        node=start
        
        while 0 in tmp:
            nodedata=string[0]
            tmp[node[0]][node[1]]=numdic[nodedata[0]]
            if node[0]-1>=0:
                tmp[node[0]-1][node[1]]=numdic[nodedata[1]]
            if node[1]+1<=m-1:
                tmp[node[0]][node[1]+1]=numdic[nodedata[2]]
            if node[0]+1<=n-1:
                tmp[node[0]+1][node[1]]=numdic[nodedata[3]]
            if node[1]-1>=0:
                tmp[node[0]][node[1]-1]=numdic[nodedata[4]]
            node=self.refreshnode(shape,node)
            string.pop(0)
        out=[]
        for i in range(n):
            outtmp=[]
            for j in range(m):
                outtmp.append(strdic[tmp[i][j]])
            out.append(outtmp)
        return out
    def refreshnode(self,shape,node):
        n=shape[0]
        m=shape[1]
        r,c=node[0],node[1]
        if m%2==0:
            for i in range(3):
                if self.dire==0:
                    if r-1>=0:
                        r=r-1
                    else:
                        c=c+1
                        self.dire=1
                elif self.dire==1:
                    if r+1<=n-2:
                        r=r+1
                    elif r+1>n-2 and r+1<=n-1 and c==m-1:
                        r=r+1
                        self.dire=2
