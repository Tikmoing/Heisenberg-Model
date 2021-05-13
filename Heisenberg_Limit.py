# -*- coding: utf-8 -*-
# author:Miyan
#date:2021.4.14

import numpy as np
import math
import random
import matplotlib.pyplot as plt

'''
L = 4
H = S1S2 + S2S3 + S3S4 + S4S1 
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
    # print(res.shape)
    for i in range(0,A.shape[0]):
        for k in range(0,A.shape[1]):
            for j in range(0,B.shape[0]):
                for p in range(0,B.shape[1]):
                    res[i*B.shape[0]+j,k*B.shape[1]+p] = A[i,k] * B[j,p] 
                    # print(i,k,j,p,i*A.shape[0]+j,k*A.shape[1]+p)
    # print(res.shape)               
    return res

def Eigvalue(A):
    res = np.linalg.eigvals(A)
    return [x.real for x in res ]

def secondMin(L):
    res = max(L)
    mi = min(L)
    for i in range(0,len(L)):
        if(res > L[i] and L[i] > mi):
            res = L[i]
    return res

# 海森堡模型
class Heisenberg():
    def __init__(self,length):
        self.length = length
       
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
    
    #得到某个量子数的不变子空间
    def getQNSpace(self,qn):
        index = self.getIndexVector(qn)
        res = np.mat([0+0j for i in range(0,len(index)**2)])
        res.shape = (len(index),len(index))
        
        for i in range(0,len(index)):
            for j in range(0,len(index)):
                res[i,j] = self.Hm[index[i],index[j]]
        
        return res
    
    #能级,必须先使用teLOP
    def getEngryList(self):
        res = []
        for i in range(0,len(self.nm)):
            res += Eigvalue(self.getQNSpace(self.nm[i]))
        res = [x.real for x in res]
        return list(set(res))
    
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
    def teLOP(self):
        self.getQuantumNumber()
        self.nm = list(set(self.quantumNumber))
        el = self.getEngryList()
        el = [round(x,9) for x in el]
        # el = list(set(el))
        # print(sorted(el))
        # print(len(el))
        E0 = min(el)
        E1 = secondMin(el)
        print("E0=%.9f"%E0,"E1=%.9f"%E1)
        return E1 - E0
        
    
h = Heisenberg(8)
h.iforUone()
h.teLOP()
# beautifulPrintMatrix(h.Hm)
# print(len([round(x.real,9) for x in np.linalg.eigvals(h.Hm)]))
res = []
L = list(range(4,13,2))
for i in L:
    h.length = i
    print("计算L=%d"%i)
    print("是否可以使用U(1)对称性:",end="")
    h.iforUone()
    dE = h.teLOP()
    print("能隙为%.9f"%dE)
    res.append(dE)
    print("\n")
res.reverse()
L = [1/x for x in L]
L.reverse()

LeE = np.polyfit(L,res,1)
print("L趋向无穷大时的一次拟合能隙为：%.9f"%LeE[1])
print("直线斜率 k = %.9f"%LeE[0],end="\n\n")
LeE = np.polyfit(L,res,2)
print("L趋向无穷大时的二次拟合能隙为：%.9f"%LeE[2])
print("二次曲线其他参数 a = %.9f , b = %.9f"%(LeE[0],LeE[1]))

plt.figure(figsize=(20,20))
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.plot(L,res)
plt.title("d E和1/L的关系",fontsize=40)
plt.xlabel("1/L",fontsize=25)
plt.ylabel("d E",fontsize=25)
plt.xticks(fontsize=24)
plt.yticks(fontsize=20)

      
