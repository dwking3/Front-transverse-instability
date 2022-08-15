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
# =============================================================================
# M_2 = 7.5
# =============================================================================
Km_1 = 5
Km_2 = 0.5
Lo_1 = 0.05 
Lo_2 = 1
lam_1 = 10## water uptake rate
lam_2 = 15
q = 5
Y_1  = 1
Y_2 = 100



## Space (X,Y) and time demensions T
X = 200
Y = 200






@njit(parallel=True,fastmath=True)
def space_integral(p,ds,B_1,B_2,W,H,k1,k2,l,m,D_W,D_H,DB_1,DB_2,M_2):
    I = np.zeros(shape=(Y,X),dtype=np.float32)
    H_T =np.zeros(shape=(Y,X),dtype=np.float32)
    W_T = np.zeros(shape=(Y,X),dtype=np.float32)
    B_1T = np.zeros(shape=(Y,X),dtype=np.float32)
    B_2T = np.zeros(shape=(Y,X),dtype=np.float32)
    Hxx,Hyy,Wxx,Wyy,B_1xx,B_1yy,B_2xx,B_2yy =np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32),np.zeros(shape=(Y,X),dtype=np.float32)
    B_1 = B_1 + k1
    B_2 = B_2 +k2
    W = W + l
    H = H +m

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
            B_2xx[i,j] = DB_2*(((B_2[ip1,j])-2*(B_2[i,j])+(B_2[im1,j]))/ds**2)
# =============================================================================
#             B_1xx[i,j] = DB_1*(((-1/12)*(B_1[im2,j]))+(4/3)*(B_1[im1,j])-(5/2)*(B_1[i,j])+(4/3)*(B_1[ip1,j])-(1/12)*(B_1[ip2,j]))/(2*ds**2) ## fourth order presicion
# =============================================================================
            B_1yy[i,j] = DB_1*(((B_1[i,jp1])-2*(B_1[i,j])+(B_1[i,jm1]))/ds**2)
            B_2yy[i,j] = DB_2*(((B_2[i,jp1])-2*(B_2[i,j])+(B_2[i,jm1]))/ds**2)
# =============================================================================
#             B_1yy[i,j] = DB_1*(((-1/12)*(B_1[i,jm2]))+(4/3)*(B_1[i,jm1])-(5/2)*(B_1[i,j])+(4/3)*(B_1[i,jp1])-(1/12)*(B_1[i,jp2]))/(2*ds**2) ## fourth order presicion
# =============================================================================
            I[i,j] = A*((Y_1*(B_1[i,j])+Y_2*(B_2[i,j])+q*f)/(Y_1*(B_1[i,j])+Y_2*(B_2[i,j])+q))
            H_T[i,j] = p - I[i,j]*(H[i,j]) +(Hxx[i,j]+Hyy[i,j])

# =============================================================================
            W_T[i,j] = I[i,j]*(H[i,j])-N_b*(W[i,j])/(1+R_b*(B_1[i,j])/Km_1) -lam_1*(W[i,j])*(B_1[i,j])*(1+E_1*(B_1[i,j]))**2 -lam_2*(W[i,j])*(B_2[i,j])*(1+E_2*(B_2[i,j]))**2+Wxx[i,j]+Wyy[i,j]

# =============================================================================
            B_1T[i,j] = Lo_1*(W[i,j])*(B_1[i,j])*(1-(B_1[i,j])/Km_1)*(1+E_1*(B_1[i,j]))**2-M_1*(B_1[i,j])+B_1xx[i,j]+B_1yy[i,j]
            B_2T[i,j] = Lo_2*(W[i,j])*(B_2[i,j])*(1-(B_2[i,j])/Km_2)*(1+E_2*(B_2[i,j]))**2-M_2*(B_2[i,j])+B_2xx[i,j]+B_2yy[i,j]
            
    return (B_1T,B_2T,W_T,H_T)
## k for B l for W m for H

## time loop

@njit(fastmath=True)
def rk_4 (Bp_1,Bp_2,W_p,H_p,p,ds,dt,D_W,D_H,DB_1,DB_2,M_2):
    B_1 = Bp_1
    B_2 = Bp_2
    W = W_p
    H = H_p
    k_O = np.zeros(shape=(Y,X),dtype=np.float32)
    k1_1,k2_1 ,l_1,m_1 = space_integral(p,ds,B_1,B_2,W,H,k_O,k_O,k_O,k_O,D_W,D_H,DB_1,DB_2,M_2)
    k1_2,k2_2, l_2,m_2 = space_integral(p,ds,B_1,B_2,W,H,0.5*dt*k1_1,0.5*dt*k2_1,0.5*dt*l_1,0.5*dt*m_1,D_W,D_H,DB_1,DB_2,M_2)
    k1_3,k2_3, l_3,m_3 = space_integral(p,ds,B_1,B_2,W,H,0.5*dt*k1_2,0.5*dt*k2_2,0.5*dt*l_2,0.5*dt*m_2,D_W,D_H,DB_1,DB_2,M_2)
    k1_4,k2_4, l_4,m_4 = space_integral(p,ds,B_1,B_2,W,H,dt*k1_3,dt*k2_3,dt*l_3,dt*m_3,D_W,D_H,DB_1,DB_2,M_2)
    B_1 = B_1 +(1/6)*dt*(k1_1+2*k1_2+2*k1_3+k1_4)
    B_2 = B_2 +(1/6)*dt*(k2_1+2*k2_2+2*k2_3+k2_4)
    W = W + (1/6)*dt*(l_1+2*l_2+2*l_3+l_4)
    H = H +(1/6)*dt*(m_1+2*m_2+2*m_3+m_4)
    return (B_1,B_2,W,H)         
                            

def tosolve (vars,*args):
    B_1 , B_2 , W , H = vars
    p ,M_2=args
    eq_1= B_1*W*Lo_1*(1-B_1/Km_1)*(1+E_1*B_1)**2-M_1*B_1
    eq_2 = B_2*W*Lo_2*(1-B_2/Km_2)*(1+E_2*B_2)**2-M_2*B_2
    eq_3 = A*(Y_1*B_1+Y_2*B_2+q*f)/(Y_1*B_1+Y_2*B_2+q)*H-N_b*W/(1+R_b*B_1/Km_1) - lam_1*W*B_1*(1+E_1*B_1)**2 - lam_2*W*B_2*(1+E_2*B_2)**2
    eq_4 = p - A*(Y_1*B_1+Y_2*B_2+q*f)/(Y_1*B_1+Y_2*B_2+q)*H 
    return [eq_1,eq_2,eq_3,eq_4]



def runner (B1_g,B2_g,W_g,H_g,p,b_X,u_X,b_Y,u_Y,k,ds,T_0,T_1,dt,D_H,D_W,DB_1,DB_2,M_2):##(B_C is newtion step first guess, p is perception rate, b_x is how many cells from half of the area the noise will begin,u_x is unitll whre,b_Y and u_Y limits in y of the noise,ds step size,k is wave number,T0 how many picturess, T1 how many steps for each picture)
    start = time.time()
# =============================================================================
# =============================================================================
#     flag='4var_c/'
#     Bp_1=np.load(flag+'B_1,P=112.5,t=200,k=0.npy')
#     Bp_2=np.load(flag+'B_2,P=112.5,t=200,k=0.npy')
#     W_p=np.load(flag+'W,P=112.5,t=200,k=0.npy')
#     H_p=np.load(flag+'H,P=112.5,t=200,k=0.npy')
#     t=201
# 
# =============================================================================
    B1_c,B2_c,W_c,H_c =fsolve(tosolve,(B1_g,B2_g,W_g,H_g),args=(p,M_2))
    print(B1_c,B2_c,W_c,H_c)
    Bp_1 =  2*B1_c*np.ones(shape=(Y,X),dtype=np.float32)
    Bp_2 =  0.2*B1_c*np.ones(shape=(Y,X),dtype=np.float32) 
    for i in range (0,Y):
        for j in range(0,X):
            if j < X/2-b_X:
                Bp_1[i,j] = Bp_1[i,j]
# =============================================================================
#                 Bp_2[i,j] = 0
# =============================================================================
            elif j > X/2-b_X-1 and j <  X/2+u_X  :
                Bp_1[i,j] = 2*0.5*Bp_1 [i,j] +Bp_1[i,j]*0.5*np.sin((k*ds)*i)
# =============================================================================
#                 Bp_2[i,j] = 2*0.5*Bp_2 [i,j] +0.5*Bp_2 [i,j]*np.sin((k*ds)*i)
# =============================================================================
            elif j > X/2+u_X-1 and j <  X/2+u_X+5  :
                    Bp_1[i,j] = 0
# =============================================================================
#                     Bp_2[i,j] = 2*0.5*Bp_2 [i,j] +0.5*Bp_2 [i,j]*np.sin((k*ds)*i)
# =============================================================================
# =============================================================================
#             elif (j > X/2-b_X-1 and j <  X/2+u_X) and ( 3*b_Y*Y<i<4*b_Y*Y or 5*b_Y*Y<i<6*b_Y*Y or 7*b_Y*Y<i<8*b_Y*Y or 9*b_Y*Y<i<10*b_Y*Y or 11*b_Y*Y<i<12*b_Y*Y or 13*b_Y*Y<i<14*b_Y*Y or 15*b_Y*Y<i<16*b_Y*Y or 17*b_Y*Y<i<18*b_Y*Y or 19*b_Y*Y<i<20*b_Y*Y)  :
#                 B[i,j] = B [i,j] 
# =============================================================================
# =============================================================================
#             elif (j > X/2-b_X-1 and j <  X/2+u_X) and ( 1.*b_Y*Y<i<3.5*b_Y*Y or  6.5*b_Y*Y<i<9*b_Y*Y )  :
#                 B[i,j] = B [i,j] 
# 
# =============================================================================
            else:
                Bp_1[i,j] = 0
# =============================================================================
#                 Bp_2[i,j] = 0
# =============================================================================
    H_p =H_c*np.ones(shape=(Y,X),dtype=np.float32) ## creating initial H with normal flactuathions mu  = 0.1 sigma =0.05
    W_p =W_c*np.ones(shape=(Y,X),dtype=np.float32)
    plt.subplot(122)
    plt.contourf(np.arange(0,X*0.5,0.5),np.arange(0,0.5*Y,0.5),Bp_2,origin=None,vmin= 0,vmax= 0.2,cmap=plt.cm.YlOrRd) ## to see the initial condithion in x-y plane
    plt.colorbar() ## the axis are not well defined
    plt.xlabel('X[m]')        
    plt.ylabel('Y[m]')
    plt.title(r'B2 $[\dfrac{kg}{m^2}]$')
    plt.subplot(121)
    plt.contourf(np.arange(0,X*0.5,0.5),np.arange(0,0.5*Y,0.5),Bp_1,origin=None,vmin= 10**-5,vmax= 1.5,cmap=plt.cm.YlGn) ## to see the initial condithion in x-y plane
    plt.colorbar() ## the axis are not well defined
    plt.xlabel('X[m]')        
    plt.ylabel('Y[m]')    
    plt.title(r'B1 $[\dfrac{kg}{m^2}]$')
    plt.suptitle('P= {:.2f} '.format(p)+r'$[\dfrac{mm}{year}]$'+r'$,M_{{2}}$={}'.format(M_2)+r'$[\dfrac{1}{year}]}$',x=0.75,y=1.05)
    plt.subplots_adjust(left=0,bottom=0.1, right=1.6, top=0.9,wspace=0.25)
    flag='4var_c/'
    t=0
    plt.savefig(flag+'zero,P={},M_2={},t={}.png'.format(p,M_2,t),bbox_inches='tight')
    plt.show()
    while t<T_0 :
        for i in repeat(None,T_1):        
            Bp_1,Bp_2,W_p,H_p = rk_4 (Bp_1,Bp_2,W_p,H_p,p,ds,dt,D_W,D_H,DB_1,DB_2,M_2)
       
        plt.subplot(122)
        plt.contourf(np.arange(0,X*0.5,0.5),np.arange(0,0.5*Y,0.5),Bp_2,origin=None,vmin= 10**-5,vmax= 1,cmap=plt.cm.YlOrRd) ## to see the initial condithion in x-y plane
        plt.colorbar() ## the axis are not well defined
        plt.xlabel('X[m]')        
        plt.ylabel('Y[m]')
        plt.title(r'B2 $[\dfrac{kg}{m^2}]$')
        plt.subplot(121)
        plt.contourf(np.arange(0,X*0.5,0.5),np.arange(0,0.5*Y,0.5),Bp_1,origin=None,vmin= 10**-5,vmax= 1.5,cmap=plt.cm.YlGn) ## to see the initial condithion in x-y plane
        plt.colorbar() ## the axis are not well defined
        plt.xlabel('X[m]')        
        plt.ylabel('Y[m]')    
        plt.title(r'B1 $[\dfrac{kg}{m^2}]$')
        plt.suptitle('P= {:.2f} '.format(p)+r'$[\dfrac{mm}{year}]$'+r'$,M_{{2}}$={}'.format(M_2)+r'$[\dfrac{1}{year}]}$',x=0.75,y=1.05)
        plt.subplots_adjust(left=0,bottom=0.1, right=1.6, top=0.9,wspace=0.25)
        flag='4var_c/'
        plt.savefig(flag+'flat ,P={},M_2={},t={}.png'.format(p,M_2,t),bbox_inches='tight')
        plt.show()
        t=t+1
    np.save(flag+'B_1,P={},t={},k={}'.format(p,t,k), Bp_1)
    np.save(flag+'B_2,P={},t={},k={}'.format(p,t,k), Bp_2)
    np.save(flag+'W,P={},t={},k={}'.format(p,t,k), W_p)
    np.save(flag+'H,P={},t={},k={}'.format(p,t,k), H_p)
    end = time.time()
    print(end-start)



for i in range(0,1):    
    runner (0.45,0.01,6,10,112.5,6,-3,0,1,0.28,0.5,100,83333,0.00032,10,1,0.1,0.05,5.5-0.1*i)