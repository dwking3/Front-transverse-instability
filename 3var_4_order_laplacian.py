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
Y = 200




## H W B_1 the variavles we look to integrate, H_T, W_T, B_T the time deteratives

@njit(parallel=True,fastmath=True)
def space_integral(p,ds,B_1,W,H,k,l,m,D_W,D_H,DB_1):
    I = np.zeros(shape=(Y,X),dtype=np.float32)
    H_T =np.zeros(shape=(Y,X),dtype=np.float32)
    W_T = np.zeros(shape=(Y,X),dtype=np.float32)
    B_T = np.zeros(shape=(Y,X),dtype=np.float32)
    Hxx,Hyy,Wxx,Wyy,B_1xx,B_1yy =np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32)
    B_1 = B_1 + k
    W = W + l
    H = H +m
# =============================================================================
#     Hx,Hy,H_laplacian = np.zeros(shape=(X,Y),dtype=np.float32),np.zeros(shape=(X,Y),dtype=np.float32),np.zeros(shape=(X,Y),dtype=np.float32)
# =============================================================================
    for i in prange (Y):
        ip1 = i+1
        im1 = i-1
        ip2 = i+2
        im2 = i-2
# =============================================================================
#         if im1<0 :##neumans
#             im1=-im1
#         if ip1  > Y-1:
#             ip1 = 2*(Y)-2-ip1
#         if im2<0 :
#             im2=-im2
#         if ip2  > Y-1:
#             ip2 = 2*(Y)-2-ip2
# =============================================================================
        if im1<0 :##periodics
            im1=Y-1
        if ip1  > Y-1:
            ip1 = 0
        if im2 < 0 :
            im2 = Y + im2
        if ip2 > Y-1 :
            ip2 = ip2 -Y
        for j in range (0,X): ## i represents y index j represnts x index
            jp1=j+1
            jm1=j-1
            jp2= j+2
            jm2 = j-2
            if jm1<0 :##neumans
            
                jm1=-jm1
            if jp1  > X-1:
                jp1 = 2*(X)-jp1-2
            if jm2<0 :
                jm2=-jm2
            if jp2  > X-1:
                jp2 = 2*(X)-2-jp2
# =============================================================================
#             if jm1<0 :##periodics
#                 jm1=X-1
#             if jp1 > X-1:
#                 jp1 = 0
#             if jm2 < 0 :
#                 jm2 = X +jm2
#             if jp2 > X-1 :
#                 jp2 = jp2 - X
# =============================================================================
# =============================================================================
#             Hx[i,j] = (H[ip1,j]-H[i,j])/(ds) ##first order
#             Hy[i,j] = (H[i,jp1]-H[i,j])/(ds)
# =============================================================================
# =============================================================================
#             Hxx[i,j]=(Hx[ip1,j]-Hx[i,j])/(ds) ##second order 1 ehtod
#             Hyy[i,j]=(Hy[i,jp1]-Hy[i,j])/(ds)
#             H_laplacian[i,j] = 2*H[i,j]*Hxx[i,j] +2*(Hx[i,j])**2+2*H[i,j]*Hyy[i,j]+2*(Hy[i,j])**2
# =============================================================================
# =============================================================================
#             Hxx[i,j] = (2*H[ip1,j]*Hx[ip1,j]-2*H[i,j]*Hx[i,j])/(ds)  ## secon order second method
#             Hyy[i,j] = (2*H[i,jp1]*Hy[i,jp1]-2*H[i,j]*Hy[i,j])/(ds) 
# # =============================================================================
# =============================================================================
#             Hxx[i,j] = D_H*(((H[ip1,j])**2-2*(H[i,j])**2+(H[im1,j])**2)/(ds**2)) ## second order normal
# =============================================================================
            Hxx[i,j] = D_H*(((-1/12)*(H[im2,j])**2)+(4/3)*(H[im1,j])**2-(5/2)*(H[i,j])**2+(4/3)*(H[ip1,j])**2-(1/12)*(H[ip2,j])**2)/(2*ds**2) ## fourth order presicion
# =============================================================================
#             Hyy[i,j] = D_H*(((H[i,jp1])**2-2*(H[i,j])**2+(H[i,jm1])**2)/(ds**2))
# =============================================================================
            Hyy[i,j] = D_H*(((-1/12)*(H[i,jm2])**2)+(4/3)*(H[i,jm1])**2-(5/2)*(H[i,j])**2+(4/3)*(H[i,jp1])**2-(1/12)*(H[i,jp2])**2)/(2*ds**2) ## fourth order presicion
            Wxx[i,j] = D_W*(((W[ip1,j])-2*(W[i,j])+(W[im1,j]))/ds**2)
# =============================================================================
#             Wxx[i,j] = D_W*(((-1/12)*(W[im2,j]))+(4/3)*(W[im1,j])-(5/2)*(W[i,j])+(4/3)*(W[ip1,j])-(1/12)*(W[ip2,j]))/(2*ds**2) ## fourth order presicion
# =============================================================================
            Wyy[i,j] = D_W*(((W[i,jp1])-2*(W[i,j])+(W[i,jm1]))/ds**2)
# =============================================================================
#             Wyy[i,j] = D_W*(((-1/12)*(W[i,jm2]))+(4/3)*(W[i,jm1])-(5/2)*(W[i,j])+(4/3)*(W[i,jp1])-(1/12)*(W[i,jp2]))/(2*ds**2) ## fourth order presicion
# =============================================================================
            B_1xx[i,j] = DB_1*(((B_1[ip1,j])-2*(B_1[i,j])+(B_1[im1,j]))/ds**2)
# =============================================================================
#             B_1xx[i,j] = DB_1*(((-1/12)*(B_1[im2,j]))+(4/3)*(B_1[im1,j])-(5/2)*(B_1[i,j])+(4/3)*(B_1[ip1,j])-(1/12)*(B_1[ip2,j]))/(2*ds**2) ## fourth order presicion
# =============================================================================
            B_1yy[i,j] = DB_1*(((B_1[i,jp1])-2*(B_1[i,j])+(B_1[i,jm1]))/ds**2)
# =============================================================================
#             B_1yy[i,j] = DB_1*(((-1/12)*(B_1[i,jm2]))+(4/3)*(B_1[i,jm1])-(5/2)*(B_1[i,j])+(4/3)*(B_1[i,jp1])-(1/12)*(B_1[i,jp2]))/(2*ds**2) ## fourth order presicion
# =============================================================================
            I[i,j] = A*((Y_1*(B_1[i,j])+q*f)/(Y_1*(B_1[i,j])+q))
            H_T[i,j] = p - I[i,j]*(H[i,j]) +(Hxx[i,j]+Hyy[i,j])

# =============================================================================
            W_T[i,j] = I[i,j]*(H[i,j])-N_b*(W[i,j])/(1+R_b*(B_1[i,j])/Km_1) -lam_1*(W[i,j])*(B_1[i,j])*(1+E_1*(B_1[i,j]))**2+Wxx[i,j]+Wyy[i,j]

# =============================================================================
            B_T[i,j] = Lo_1*(W[i,j])*(B_1[i,j])*(1-(B_1[i,j])/Km_1)*(1+E_1*(B_1[i,j]))**2-M_1*(B_1[i,j])+B_1xx[i,j]+B_1yy[i,j]
            
    return (B_T,W_T,H_T)
## k for B l for W m for H

## time loop

@njit(fastmath=True)
def rk_4 (B,W_p,H_p,p,ds,dt,D_W,D_H,DB_1):
    B_1 = B
    W = W_p
    H = H_p
    k_O = np.zeros(shape=(Y,X),dtype=np.float32)
    l_O = np.zeros(shape=(Y,X),dtype=np.float32)
    m_O = np.zeros(shape=(Y,X),dtype=np.float32)
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
#     flag='longnonlinear/'
#     t=480
#     B=np.load(flag+'B,P=119,t=480,k=0.56.npy')
#     W_p=np.load(flag+'W,P=119,t=480,k=0.56.npy')
#     H_p=np.load(flag+'H,P=119,t=480,k=0.56.npy')
# =============================================================================
    B_c,H_c,W_c = inisolver(p,B_C)
    print(B_c,H_c,W_c)
    B =  B_c*np.ones(shape=(Y,X),dtype=np.float32)
    for i in range (0,Y):
        for j in range(0,X):
            if j < X/2-b_X:
                B[i,j] = B[i,j]
            elif j > X/2-b_X-1 and j <  X/2+u_X  :
                B[i,j] = 0.5*B [i,j] +B[i,j]*0.5*np.sin((k*ds)*i)

            elif (j > X/2-b_X-1 and j <  X/2+u_X) and ( 3*b_Y*Y<i<4*b_Y*Y or 5*b_Y*Y<i<6*b_Y*Y or 7*b_Y*Y<i<8*b_Y*Y or 9*b_Y*Y<i<10*b_Y*Y or 11*b_Y*Y<i<12*b_Y*Y or 13*b_Y*Y<i<14*b_Y*Y or 15*b_Y*Y<i<16*b_Y*Y or 17*b_Y*Y<i<18*b_Y*Y or 19*b_Y*Y<i<20*b_Y*Y)  :
                B[i,j] = B [i,j] 
            elif (j > X/2-b_X-1 and j <  X/2+u_X) and ( 1.*b_Y*Y<i<3.5*b_Y*Y or  6.5*b_Y*Y<i<9*b_Y*Y )  :
                B[i,j] = B [i,j] 

            else:
                B[i,j] = 0
    H_p =H_c*np.ones(shape=(Y,X),dtype=np.float32) ## creating initial H with normal flactuathions mu  = 0.1 sigma =0.05
    W_p =W_c*np.ones(shape=(Y,X),dtype=np.float32)
    t=0
    plt.contourf(np.arange(0,X*0.5,0.5),np.arange(0,0.5*Y,0.5),B,origin=None,cmap=plt.cm.YlGn) ## to see the initial condithion in x-y plane
    plt.colorbar(label=r'B1 $[\dfrac{kg}{m^2}]$') ## the axis are not well defined
    plt.xlabel('X[m]')        
    plt.ylabel('Y[m]')
    plt.title('P= {:.2f} '.format(p)+r'$[\dfrac{mm}{year}]$')
    flag='longnonlinear/'
    plt.savefig(flag+'1,P={},t={}.png'.format(p,t))
    plt.show()
    while t<T_0+480 :
        for i in repeat(None,T_1):        
            B,W_p,H_p = rk_4 (B,W_p,H_p,p,ds,dt,D_W,D_H,DB_1)
        plt.contourf(np.arange(0,X*0.5,0.5),np.arange(0,0.5*Y,0.5),B,origin=None,cmap=plt.cm.YlGn) ## to see the initial condithion in x-y plane
        plt.colorbar(label=r'B2 $[\dfrac{kg}{m^2}]$') ## the axis are not well defined
        plt.xlabel('X[m]')        
        plt.ylabel('Y[m]')
        plt.title('P= {:.2f} '.format(p)+r'$[\dfrac{mm}{year}]$')
        flag='longnonlinear/'
        plt.savefig(flag+'1,P={},t={}.png'.format(p,t))
        plt.show()
        t=t+1
    end = time.time()
    print(end-start)
    print(B)
    np.save(flag+'B,P={},t={},k={}'.format(p,t,k), B)
    np.save(flag+'W,P={},t={},k={}'.format(p,t,k), W_p)
    np.save(flag+'H,P={},t={},k={}'.format(p,t,k), H_p)


for i in range(1,2 ):    
    runner (0.45,112,60,-17,0,1,0.28,0.5,160,83333,0.00032,10,1,0.1)