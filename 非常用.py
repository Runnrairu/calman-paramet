# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt


def a(theta):
    return 3*theta+0.01

def b(theta):
    return theta*theta

def c(theta):
    return 2
    
def r(theta):
    x = a(theta)*a(theta)+b(theta)*b(theta)*c(theta)*c(theta)/(sigma_o*sigma_o)
    return pow(x,1/2)

def gamma(theta):
    return r(theta)-a(theta)

def m_t_y(y):
    m_y = [0]*(m+1)
    
    r_y=r(y)
    a_y=a(y)
    gamma_y = r_y-a_y
    m_y[0]=np.random.normal(0,pow(gamma_y*gamma_y*sigma_o*sigma_o/(2*a_y),0.5))
    for i in range(m):
        #e_r = pow(e,-float(i)/t*r_y)
        #m_y_etr[i+1] = m_y_etr[i]+pow(e,float(i)/t*r_y)*gamma_y*(Y[i+1]-Y[i])
        #m_y[i+1] = m_y_etr[i+1]*e_r
        m_y[i+1] = m_y[i]-(a_y+gamma_y)*delta_t+gamma_y*(Y[i+1]-Y[i])
    return m_y
    

def L(y):#尤度関数を書く 
    m_y=m_t_y(y)
    sumx=0
    sumt=0
    
    for i in range(m):      
        sumx += m_y[i+1]*(Y[i+1]-Y[i])
        sumt += m_y[i+1]*m_y[i+1]*delta_t  
    return sumx/(sigma_o*sigma_o)-sumt/(2*sigma_o*sigma_o)

def p(y):#事前分布
    if y<1 and y>0:
        return 1
    else:
        return 0

def q(y):#事後分布（定数除く）
    return p(y)*L(y)


e=2.71828182846
 #時刻は1000分割、100秒
T=20.0
t=1000#分割係数
n=int(T)*t
theta=0.5#random.random()#[0,1]の一葉乱数
X =[0]*(n+1)
Y =[0]*(n+1)
delta_t = 1.0/t
theta1=0.5
sigma = np.power(delta_t,0.5)
sigma_o =1.0 #最初にわかってる分散
a_theta = a(theta)
b_theta = b(theta)
c_theta = c(theta)
r_theta = r(theta)
a_theta1 = a(theta1)
b_theta1 = b(theta1)
c_theta1 = c(theta1)
r_theta1 = r(theta1)
gamma_theta1 = r_theta1-a_theta1
print(theta)
gamma_theta = gamma(theta)
dW_y =np.random.normal(0,sigma,n)
dW_x =np.random.normal(0,sigma,n)

m_theta1 = [0]*(n+1)
#sample = 10000#サンプル数
#MC = [[0 for i in range(sample)]for j in range(int(T))] 


for i in range(n):
    deltaX=-a_theta*X[i]*delta_t+b_theta*dW_x[i]
    deltaY=c_theta*X[i]*delta_t + sigma_o*dW_y[i]
    X[i+1] = X[i]+deltaX
    Y[i+1] = Y[i]+deltaY
    m_theta1[i+1] = m_theta1[i]-(a_theta1+gamma_theta1)*m_theta1[i]*delta_t+gamma_theta1*deltaY

maxim = [0]*(int(T))

for i in range(int(T)):#T/10秒ご
    
    m = (i+1)*t
    y_i=np.array([0 for j in range(100)])
    print(m)
    for j in range(100):
        
        y_i[j] = L(float(j)/100)
        
    maxim[i] = np.argmax(y_i)
plt.plot(maxim)
plt.show()
    
    


