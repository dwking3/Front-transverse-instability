import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from numba import njit, prange
import time
from itertools import repeat

## model parameters block
f = 0.2 #contrast
A = 40
N_b = 30
R_b = 30
E_1 = 2 ## root to shoot
E_2 = 0
M_1 = 0.75
M_2 = 7.5
Km_1 = 5
Km_2 = 0.5
Lo_1 = 0.05 
Lo_2 = 1
lam_1 = 10## water uptake rate
lam_2 = 15
q = 5
Y_1  = 1
Y_2 = 100
# =============================================================================
# DB_1 = 0.1
# =============================================================================
DB_2 = 0.05



## Space (X,Y) and time demensions T
X = 200




## H W B_1 the variavles we look to integrate, H_T, W_T, B_T the time deteratives

@njit(parallel=True,fastmath=True)
def space_integral(p,ds,B_1,W,H,k,l,m,D_W,D_H,DB_1):
    I = np.zeros(shape=(X),dtype=np.float32)
    H_T =np.zeros(shape=(X),dtype=np.float32)
    W_T = np.zeros(shape=(X),dtype=np.float32)
    B_T = np.zeros(shape=(X),dtype=np.float32)
    Hxx,Wxx,B_1xx =np.zeros(shape=(X),dtype=np.float32),np.zeros(shape=(X),dtype=np.float32),np.zeros(shape=(X),dtype=np.float32)
    B_1 = B_1 + k
    W = W + l
    H = H +m

    for i in prange (X):
        ip1 = i+1
        im1 = i-1
        ip2 = i+2
        im2 = i-2
        if im1<0 :##neumans
            im1=-im1
        if ip1  > X-1:
            ip1 = 2*(X)-2-ip1
        if im2<0 :
            im2=-im2
        if ip2  > X-1:
            ip2 = 2*(X)-2-ip2
# =============================================================================
#         if im1<0 :##periodics
#             im1=X-1
#         if ip1  > X-1:
#             ip1 = 0
#         if im2 < 0 :
#             im2 = X + im2
#         if ip2 > X-1 :
#             ip2 = ip2 -X
# =============================================================================
        
# =============================================================================
#             Hxx[i] = D_H*(((H[ip1])**2-2*(H[i])**2+(H[im1])**2)/(ds**2)) ## second order normal
# =============================================================================
        Hxx[i] = D_H*(((-1/12)*(H[im2])**2)+(4/3)*(H[im1])**2-(5/2)*(H[i])**2+(4/3)*(H[ip1])**2-(1/12)*(H[ip2])**2)/(2*ds**2) ## fourth order presicion
# =============================================================================

        Wxx[i] = D_W*(((W[ip1])-2*(W[i])+(W[im1]))/ds**2)
# =============================================================================
#             Wxx[i,j] = D_W*(((-1/12)*(W[im2,j]))+(4/3)*(W[im1,j])-(5/2)*(W[i,j])+(4/3)*(W[ip1,j])-(1/12)*(W[ip2,j]))/(2*ds**2) ## fourth order presicion
# =============================================================================
        B_1xx[i] = DB_1*(((B_1[ip1])-2*(B_1[i])+(B_1[im1]))/ds**2)
# =============================================================================
#             B_1xx[i,j] = DB_1*(((-1/12)*(B_1[im2,j]))+(4/3)*(B_1[im1,j])-(5/2)*(B_1[i,j])+(4/3)*(B_1[ip1,j])-(1/12)*(B_1[ip2,j]))/(2*ds**2) ## fourth order presicion
# =============================================================================
            
        I[i] = A*((Y_1*(B_1[i])+q*f)/(Y_1*(B_1[i])+q))
        H_T[i] = p - I[i]*(H[i]) +(Hxx[i])

# =============================================================================
        W_T[i] = I[i]*(H[i])-N_b*(W[i])/(1+R_b*(B_1[i])/Km_1) -lam_1*(W[i])*(B_1[i])*(1+E_1*(B_1[i]))**2+Wxx[i]

# =============================================================================
        B_T[i] = Lo_1*(W[i])*(B_1[i])*(1-(B_1[i])/Km_1)*(1+E_1*(B_1[i]))**2-M_1*(B_1[i])+B_1xx[i]
            
    return (B_T,W_T,H_T)
## k for B l for W m for H

## time loop

@njit(fastmath=True)
def rk_4 (B,W_p,H_p,p,ds,dt,D_W,D_H,DB_1):
    B_1 = B
    W = W_p
    H = H_p
    k_O = np.zeros(shape=(X),dtype=np.float32)
    l_O = np.zeros(shape=(X),dtype=np.float32)
    m_O = np.zeros(shape=(X),dtype=np.float32)
    k_1, l_1,m_1 = space_integral(p,ds,B_1,W,H,k_O,l_O,m_O,D_W,D_H,DB_1)
    k_2, l_2,m_2 = space_integral(p,ds,B_1,W,H,0.5*dt*k_1,0.5*dt*l_1,0.5*dt*m_1,D_W,D_H,DB_1)
    k_3, l_3,m_3 = space_integral(p,ds,B_1,W,H,0.5*dt*k_2,0.5*dt*l_2,0.5*dt*m_2,D_W,D_H,DB_1)
    k_4, l_4,m_4 = space_integral(p,ds,B_1,W,H,dt*k_3,dt*l_3,dt*m_3,D_W,D_H,DB_1)
    B_1 = B_1 +(1/6)*dt*(k_1+2*k_2+2*k_3+k_4)
    W = W + (1/6)*dt*(l_1+2*l_2+2*l_3+l_4)
    H = H +(1/6)*dt*(m_1+2*m_2+2*m_3+m_4)
    return (B_1,W,H)         
                            

def tosolve (B,p): ##function defenition in order to work with fsolve
    root=(B*p/(N_b/(1+R_b*B/Km_1) + lam_1*B*(1+E_1*B)**2))*Lo_1*(1-B/Km_1)*(1+E_1*B)**2-M_1*B ## equathion to solve after all the algebra done
    return (root)
def inisolver (p,B_c):
    B =fsolve(tosolve,(B_c),p) ## sole B for to differnt gueses (0,3)
    H_t=p*(Y_1*B+q)/(A*(Y_1*B+q*f)) ## the value of W using B, the eqution is written after all the algebra was done
    W_t = p/(N_b/(1+R_b*B/Km_1) + lam_1*B*(1+E_1*B)**2)
    return(B,H_t,W_t)


def runner (B_C,p,b_X,u_X,b_Y,u_Y,k,ds,T_0,T_1,dt,D_H,D_W,DB_1):##(B_C is newtion step first guess, p is perception rate, b_x is how many cells from half of the area the noise will begin,u_x is unitll whre,b_Y and u_Y limits in y of the noise,ds step size,k is wave number,T0 how many picturess, T1 how many steps for each picture)
    start = time.time()
# =============================================================================
#     V=np.zeros(shape=(X),dtype=np.float32)
# =============================================================================
    B_c,H_c,W_c = inisolver(p,B_C)
    print(B_c,H_c,W_c)
    B =  B_c*np.ones(shape=(X),dtype=np.float32)
    for i in range (0,X):
        if i < X/2+1:
            B[i] = B[i]
        else:
            B[i] = 0
    H_p =H_c*B_c*np.ones(shape=(X),dtype=np.float32) ## creating initial H with normal flactuathions mu  = 0.1 sigma =0.05
    W_p =W_c* B_c*np.ones(shape=(X),dtype=np.float32)
    t=0
    X_m = X/2 ##for start
    while t<T_0 : 
        for i in repeat(None,T_1):        
            B,W,H = rk_4 (B,W_p,H_p,p,ds,dt,D_W,D_H,DB_1)
            H_p = H
            W_p = W
        plt.plot(np.arange(0,X/2,0.5),B)
        plt.xlabel('X')        
        plt.ylabel('B')    
        plt.title(' P= {}'.format(p))
        plt.show()
# =============================================================================
#         X_p =np.argmax(B[15:X-15])
# =============================================================================
        t=t+1
# =============================================================================
#         V[i]= ds*(X_p-X_m)/(T_1*dt)
# =============================================================================
# =============================================================================
#         X_m = X_p
# =============================================================================
    end = time.time()
    X_p =np.argmax(B[10:X-10])+10
# =============================================================================
#     print(X_p)
# =============================================================================
    V_avg = ds*(X_p-X_m)/(T_1*T_0*dt)
    print(V_avg)
    plt.plot(np.arange(0,X/2,0.5),B)
    plt.xlabel('X')        
    plt.ylabel('B')    
    plt.title(' P= {}'.format(p))
    plt.show()
# =============================================================================
#     print(end-start)
# =============================================================================
# =============================================================================
#     V_avg = np.average(V[1:])
# =============================================================================
    return (V_avg)
# =============================================================================
# print(runner (0.28125271,71.03,-60,67,0.1,0.85,0.24,0.5,10,20000,0.00032,10,1,0.1))
# =============================================================================

def velocitygraph (B_C,p,b_X,u_X,b_Y,u_Y,k,ds,T_0,T_1,dt,D_H,D_W,DB_1,n):
    V=np.zeros(n)
    pgraph = np.zeros(n)
    for i in range (0,n):
        p_c = p+0.4*i
        pgraph[i] = p_c
        V[i]=runner(B_C,p_c,b_X,u_X,b_Y,u_Y,k,ds,T_0,T_1,dt,D_H,D_W,DB_1)
    Verr = np.sqrt(2)*(ds/(T_1*T_0*dt))
    plt.plot(pgraph,V)
    plt.errorbar(pgraph, V, yerr = Verr)
    plt.xlabel('p')        
    plt.ylabel('v')    
    plt.title('V(p)')
    plt.show()

velocitygraph (0.6,110.5,-60,67,0.1,0.85,0.24,0.5,5,200000,0.00032,10,1,0.1,43)
