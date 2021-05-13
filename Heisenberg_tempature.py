# -*- coding: utf-8 -*-
# author:Miyan
#date:2021.4.19

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import math

'''
L = 8时海森堡模型的比热
'''

#打印矩阵
def beautifulPrintMatrix(A):
    n = A.shape[0]
    for i in range(0,n):
        for j in range(0,n):
            oe = A[i,j]
            print("%.9f "%(oe.real),end="  ")
        print("")

#向量直乘
def Stv(a,b):
    la = a.shape[0]
    lb = b.shape[0]
    res = np.array([0 for i in range(0,la*lb)])
    
    for i in range(0,la):
        for j in range(0,lb):
            res[j + i * lb] = a[i]+b[j]
    
    return res

#矩阵直乘
def Stm(A,B):
    shapeA = A.shape
    shapeB = B.shape
    rowRes = shapeA[0]*shapeB[0]
    colRes = shapeA[1]*shapeB[1]
    res = np.mat([0+0j for i in range(0,rowRes*colRes)])
    res.shape = (rowRes,colRes)

    for i in range(0,A.shape[0]):
        for k in range(0,A.shape[1]):
            for j in range(0,B.shape[0]):
                for p in range(0,B.shape[1]):
                    res[i*B.shape[0]+j,k*B.shape[1]+p] = A[i,k] * B[j,p] 
    return res

def Eig(A):
    vv = np.linalg.eig(A)
    for i in range(0,len(vv[0])):
    	vv[0][i] = round(vv[0][i].real,9)

    return vv


def sortai(A):
	for i in range(0,len(A)):
		for j in range(0,len(A)-i-1):
			if(A[j][0] < A[j+1][0]):
				A[j][0],A[j+1][0] = A[j+1][0],A[j][0]

	return A



# 海森堡模型
class Heisenberg():
    def __init__(self,length,temperature):
        self.length = length
       	self.temperature = temperature
        #自旋为1/2的矩阵
        self.eye = np.mat(np.eye(2,dtype = complex))
        self.spx = np.mat(np.array([[0,1/2+0j],[1/2,0]]))
        self.spy = np.mat(np.array([[0,-0.5j],[0.5j,0]]))
        self.spz = np.mat(np.array([[1/2+0j,0],[0,-1/2+0j]]))
        self.spv = [self.spx,self.spy,self.spz]

    
    def getSzMatrix(self,m):
        res = np.mat([1+0j])
        for i in range(1,self.length+1):
            if(i != m ):
                 res = Stm(res,self.eye)
            else:
                 res = Stm(res,self.spz)
        return res        

	#m和n物体dist方向自旋态乘积        
    def getMatrixWith(self,dist,m,n):
        res = np.mat([1+0j])
        for i in range(1,self.length+1):
            if(i != m and i != n):
                 res = Stm(res,self.eye)
            else:
                 res = Stm(res,self.spv[dist])
        return res
	
	#m和n物体自旋内积矩阵   
    def getSdotSMatrixWith(self,m,n):
        res = self.getMatrixWith(0,m,n) + self.getMatrixWith(1,m,n) + self.getMatrixWith(2,m,n)
        return res
	
	#海森堡模型矩阵
    def getHeisenbergMatrixPBC(self):
        if(self.length == 2):
            self.Hm = self.getSdotSMatrixWith(1,self.length)
            return self.Hm
        res = self.getSdotSMatrixWith(1,self.length)
        for i in range(1,self.length):
            res += self.getSdotSMatrixWith(i,i+1)
        self.Hm = res
        return res
    
    #计算 \sum{S_z}
    def getSumSz(self):
        res = self.getSzMatrix(1)
        for i in range(2,self.length+1):
            res += self.getSzMatrix(i)
        self.Szm = res
        return res
    
    #得到量子数
    def getQuantumNumber(self):
        temp = np.array([1,-1])
        res = np.array([1,-1])
        for i in range(1,self.length):
            res = Stv(res,temp)
        self.quantumNumber = res.tolist()
        return res
    
    #得到某一个量子数的对应索引
    def getIndexVector(self,qn):
        index = []
        for i in range(0,len(self.quantumNumber)):
            if(self.quantumNumber[i] == qn):
                index.append(i)
        return index
    
    #得到某个量子数的不变子空间和本征值
    def getQNSpace(self,qn):
        index = self.getIndexVector(qn)
        res = np.mat([0+0j for i in range(0,len(index)**2)])
        res.shape = (len(index),len(index))
        
        for i in range(0,len(index)):
            for j in range(0,len(index)):
                res[i,j] = self.Hm[index[i],index[j]]
        temp = Eig(res)
        values = (temp[0]).tolist()
        values = [x.real for x in values]
        vector = []
        # 因为得到的本征向量维数小于原哈密顿矩阵故将其扩维
        for i in range(0,len(index)):
            p = temp[1][:,i]
            t = []
            cout = 0
            for j in range(0,2**self.length):
                if(j in index):
                    t.append(p[cout][0,0])
                    cout += 1
                else:
                    t.append(0)
            vector.append(np.array(t))
                
        return [values,vector]
    
        #判断是否能使用U(1)对称性
    def iforUone(self):
        self.getHeisenbergMatrixPBC()
        self.getSumSz()
        
        p = self.Hm * self.Szm - self.Szm * self.Hm
        res = 0
        for i in range(0,p.shape[0]):
            for j in range(0,p.shape[1]):
                res += abs(p[i,j])
        if(res<= 0.001):
            print("Yes")
            return 1
        else:
            print("No")
            return 0

    #进行对角化
    def diagonalizationH(self):
        self.getQuantumNumber()
        self.nm = list(set(self.quantumNumber))
        eigVectorRes = []
        eigValueRes = []
        for i in self.nm:
            temp = self.getQNSpace(i)
            eigValueRes = eigValueRes +  temp[0]
            eigVectorRes = eigVectorRes + temp[1]

        diag = np.mat([0.0 for i in range(0,2**self.length*2**self.length)])
        par = np.mat([0.0+0.0j for i in range(0,2**self.length*2**self.length)],dtype=complex)
        diag.shape = (2**self.length,2**self.length)
        par.shape = (2**self.length,2**self.length)
        
        for i in range(0,2**self.length):
            for j in range(0,2**self.length):
                if( i == j):
                    diag[i,j] = eigValueRes[i]
                par[i,j] = eigVectorRes[i][j]
        
        par = par.T
        self.dr = [diag,par]

        return [diag,par]

    def getZH(self):
        ediag = np.mat([0.0 for i in range(0,2**self.length*2**self.length)])
        ediag.shape = (2**self.length,2**self.length)

        for i in range(0,2**self.length):
            ediag[i,i] = math.exp(-self.dr[0][i,i] / self.temperature)

        ediag = self.dr[1] * ediag *  (self.dr[1]).I
        Hediag = self.Hm * ediag
        
        trace = 0
        Htrace = 0
        for i in range(0,2**self.length):
            trace += ediag[i,i]
            Htrace += Hediag[i,i]
        return [trace.real,Htrace.real,ediag,Hediag]

    def getCv(self):
        tau = self.getZH()
        self.temperature = self.temperature + 0.000001
        tau1 = self.getZH() 
        
        return (tau1[1]/tau1[0] - tau[1]/tau[0]) / 0.000001 / self.length


h4 = Heisenberg(4,0)
h6 = Heisenberg(6,0)
h8 = Heisenberg(8,0)
h10 = Heisenberg(10,0)

print("L = 4 是否可以应用U(1)对称性：",end="")
h4.iforUone()
h4.diagonalizationH()

print("L = 6 是否可以应用U(1)对称性：",end="")
h6.iforUone()
h6.diagonalizationH()

print("L = 8 是否可以应用U(1)对称性：",end="")
h8.iforUone()
h8.diagonalizationH()

print("L = 10 是否可以应用U(1)对称性：",end="")
h10.iforUone()
h10.diagonalizationH()

res4 = []
res6 = []
res8 = []
res10 = []
T = []
for i in range(0,81):
    h4.temperature = 0.01 + i * 0.025
    h6.temperature = 0.01 + i * 0.025
    h8.temperature = 0.01 + i * 0.025
    h10.temperature = 0.01 + i * 0.025
    T.append(h4.temperature)
    res4.append(h4.getCv())
    res6.append(h6.getCv())
    res8.append(h8.getCv())
    res10.append(h10.getCv())

plt.figure(figsize=(15,15))
plt.plot(T,res4,linewidth=3.5)
plt.plot(T,res6,linewidth=3.5)
plt.plot(T,res8,c = 'r',linewidth=3.5)
plt.plot(T,res10,linewidth=3.5)

plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.legend(["L = 4","L = 6","L = 8","L = 10"],fontsize=25)
plt.title("Cv 与 T 在不同L下的曲线",fontsize=40)
plt.xlabel("T",fontsize=25)
plt.ylabel("Cv",fontsize=25)
plt.xticks(fontsize=24)
plt.yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45],fontsize=20)