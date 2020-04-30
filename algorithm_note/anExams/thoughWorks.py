import numpy as np

class UnmannedAerialVehicle():
    def __init__(self):
        self.dire=0    #0,1,2: up,down,left
        # self.dirlis=[0,1,2]
    
    # 重建地形
    def buildmap(self,shape,string):
        '''
            shape:地图形状
            sting:字符串
        '''
        n=shape[0]
        m=shape[1]
        numdic={'G':1,'F':2,'R':3, '-':4, 'f':0}
        strdic={1:'G',2:'F',3:'R', 4:'-', 0:'f'}
        tmp=np.zeros((n,m), dtype=np.int)
        start=np.array([n-2,0])
        node=start
        
        # 当0时，还有没有复原的位置，仅需更新
        while 0 in tmp:
            if len(string)==0:
                break
            else:

                nodedata=string[0]
                tmp[node[0]][node[1]]=numdic[nodedata[0]]
                # 是否超出边界
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
        print(tmp)
        # tmp数字转换字符串
        out=[]
        for i in range(n):
            outtmp=[]
            for j in range(m):
                outtmp.append(strdic[tmp[i][j]])
            out.append(outtmp)
        return out
    
    # 往下走走
    def refreshnode(self,shape,node):
        '''
            shape:地图形状
            node:当前位置节点
        '''
        n=shape[0]
        m=shape[1]
        r,c=node[0],node[1]
        if m%2==0 or m%2==1:
            for i in range(1):
                if self.dire==0:
                    if r-1>0:
                        r=r-1
                    else:
                        c=c+1
                        self.dire=1
                elif self.dire==1:
                    if r+1<=n-2:
                        r=r+1
                    # elif r+1>n-2 and r+1<=n-1 and c==m-1:
                    else:
                        c=c+1
                        self.dire=2
                else:
                    while c>=0:
                    c = c-1




        return [r, c]

if __name__ == "__main__":
    UAV = UnmannedAerialVehicle()
    shape=[4,4]
    detect=[['G','F','R','G','-'],['G','-','R','F','R'],['G','F','R','F','R'],['G','-','-','F','R'],['G','R','-','-','F'],['G','G','R','-','-']]
    res1= UAV.buildmap(shape, detect)
    print(res1)
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