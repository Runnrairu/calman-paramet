# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def a(theta):
    return theta/10

def b(theta):
    return 0.2

def c(theta):
    return 0.05
    
def r(theta):
    x = a(theta)*a(theta)+b(theta)*b(theta)*c(theta)*c(theta)/(sigma_o*sigma_o)
    return pow(x,1/2)

def gamma(theta):
    return r(theta)-a(theta)

def m_t_y(y,Y,i,T):
    
    
    r_y=r(y)
    a_y=a(y)
    gamma_y = r_y-a_y
    step=int(T*nt)
    m_y = [0]*(step+1)
    for t in range(step):
        
        m_y[t+1] = m_y[t]+(gamma_y-a_y)*m_y[t]+gamma_y*(Y[i][t+1]-Y[i][t])
    return m_y

def L(y,Y,i,T):#対数尤度関数を書く 
    m_y=m_t_y(y,Y,i,T+1)#m_yの計算
    sumx=0
    sumt=0
    step = int((T+1)*nt)
    for t in range(step):      
        sumt +=  (m_y[t]*m_y[t]-m_theta1[i][t]*m_theta1[i][t])*delta_t
        sumx +=  (m_y[t]-m_theta1[i][t])*(Y[i][t+1]-Y[i][t])
        
    return sumx/(step*sigma_o*sigma_o)-sumt/(step*2*sigma_o*sigma_o)



e=2.71828182846
 #時刻は1秒1000分割、10秒
T=10.0
nt=1000#分割係数
n=int(T)*nt
test_case = 1000 #いくつサンプルパスを作るか
X =[[0 for t in range(n+1)]for i in range(test_case)]
Y =[[0 for t in range(n+1)]for i in range(test_case)]
delta_t = 1.0/nt
theta=np.random.rand()+0.5#0.5~1.5の乱数
sigma = np.power(delta_t,0.5)#ブラウン運動の分散
sigma_o =1.0 #最初にわかってる分散
a_theta = a(theta)
b_theta = b(theta)
c_theta = c(theta)
r_theta = r(theta)
gamma_theta= r_theta-a_theta
print(theta)
theta1=1.0
a_theta1 = a(theta1)
b_theta1 = b(theta1)
c_theta1 = c(theta1)
r_theta1 = r(theta1)
gamma_theta1= r_theta1-a_theta1
dW_y =np.random.normal(0,sigma,(test_case,n))
dW_x =np.random.normal(0,sigma,(test_case,n))

m_theta1 = [[0 for i in range(int(T))]for t in range(test_case)]



for i in range(test_case):
    n_theta1 = [0]*(n+1)
    for t in range(n):
        deltaX=-a_theta*X[i][t]*delta_t+b_theta*dW_x[i][t]
        deltaY=c_theta*X[i][t]*delta_t + sigma_o*dW_y[i][t]
        X[i][t+1] = X[i][t]+deltaX
        Y[i][t+1] = Y[i][t]+deltaY
        n_theta1[t+1] = n_theta1[t]+pow(e,float(t/nt)*r_theta1)*deltaY
    m_theta1[i]=m_t_y(theta1,Y,i,T)
cutT=5
while(cutT<(int(T)-1)):
    opt_theta = [0]*(1001)#区間を1000分し、カウントする
    for i in range(test_case):
        y=0.5
        
        maxL=-100000000000000#Lの最大値。とにかく小さくとる
        miny=-1 #argmax_y L 
        for j in range(1001):#区間の1000分割
            
            y += 1.0/1000
            L_y=L(y,Y,i,cutT+1)
            if maxL < L_y:
                miny= j
                maxL= L_y
        opt_theta[miny] += 1
    plt.plot(opt_theta)
    cutT += 4
        
    
    
    



plt.show()
    
    


