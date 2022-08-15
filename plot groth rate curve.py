import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.matrices import Matrix
from sympy.core.symbol import symbols
B, W, H,k,X,p= symbols('B, W, H,k,X,p ', real=True)
from scipy.optimize import fsolve
from scipy import optimize
from numba import njit, prange
f = 0.2 #contrast
A = 40
N_b = 30
R_b = 30
# =============================================================================
# N_a =2
# R_a =120
# =============================================================================
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
DB_1 =0.1
D_H=10
D_W=1
eq_1= B*W*Lo_1*(1-B/Km_1)*(1+E_1*B)**2-M_1*B
eq_2 = A*(Y_1*B+q*f)/(Y_1*B+q)*H-N_b*W/(1+R_b*B/Km_1) - lam_1*W*B*(1+E_1*B)**2
eq_3 = p - A*(Y_1*B+q*f)/(Y_1*B+q)*H 
# =============================================================================
# -N_a*H/(1+R_a*B/Km_1)
# =============================================================================
d1B = simplify(diff(eq_1, B))
d1W = simplify(diff(eq_1, W))
d1H = simplify(diff(eq_1, H))
d2B = simplify(diff(eq_2, B))
d2W = simplify(diff(eq_2, W))
d2H = simplify(diff(eq_2, H))
d3B = simplify(diff(eq_3, B))
d3W = simplify(diff(eq_3, W))
d3H = simplify(diff(eq_3, H))
jac_0 = Matrix(([d1B-DB_1*k**2-X,d1W,d1H],[d2B,d2W-D_W*k**2-X,d2H],[d3B,d3W,d3H-2*H*D_H*k**2-X]))
def tosolve (B,p):
    root=(B*p/(N_b/(1+R_b*B/Km_1) + lam_1*B*(1+E_1*B)**2))*Lo_1*(1-B/Km_1)*(1+E_1*B)**2-M_1*B
    return (root)
def extract_coefficients(u, d):
    return (np.linalg.solve([[[x for x in range(d+1)][line] ** column for column in range(d+1)] for line in range(d+1)], [u(x) for x in [x for x in range(d+1)]]))[::-1]

def graphfinder(p_t,n,tempmatrix):
    p = p_t
    B_t =fsolve(tosolve,([0.3,0.8]),p_t)[1]
    print(B_t)
    H_t=p/((A*(Y_1*B+q*f)/(Y_1*B+q)))
    W_t = p/(N_b/(1+R_b*B/Km_1) + lam_1*B*(1+E_1*B)**2)
    jac1 =simplify( tempmatrix.subs(H,H_t))
    jac2 = simplify(jac1.subs(W,W_t))
    jac_3 = simplify(jac2.subs(B,B_t))
    evalue = np.zeros(n)        
    for j in range(0,n):
        k_t=0+0.01*j  
        jac_4 = simplify(jac_3.subs(k,k_t))
        lam_f = lambdify(X,(jac_4.det()))
        coef = extract_coefficients(lam_f,3)
        evalue[j] = np.roots(coef)[2]
    return (evalue)
# =============================================================================
# print(graphfinder(119,100,jac_0))
# =============================================================================
for j in range(0,20):   
    for i in range(0,3) :
        p=111.1+0.1*i+0.3*j
        f=0.1
        if i ==0:
    # =============================================================================
            ax1 = plt.subplot(311)
        # =============================================================================
            temp = (graphfinder(p,500,jac_0))
            plt.plot(np.arange(0,5,0.01),temp,label ='P= {:.2f}'.format(p))
        # =============================================================================
        #     plt.tick_params('x', labelsize=6)
        # =============================================================================
                   
               
            plt.title('growth-rate curve')
            plt.axhline(y=0.0, color='k', linestyle='--')
            plt.axis([0, 2, -0.5, 0.5])
            plt.scatter(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp), color='red')
            plt.text(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp),'({:.2f},{:.2f})'.format(*np.arange(0,5,0.01)[np.where(temp==np.max(temp))],np.max(temp)),horizontalalignment='left',verticalalignment='bottom')
            plt.tick_params('x', labelbottom=False)
            plt.legend()
    # =============================================================================
        if i ==1:
            ax2 = plt.subplot(312, sharex=ax1,sharey=ax1)
            temp = (graphfinder(p,500,jac_0))
            plt.plot(np.arange(0,5,0.01),temp,label ='P= {:.2f}'.format(p))         
            plt.axhline(y=0.0, color='k', linestyle='--')
            plt.axis([0, 2, -0.5, 0.5])
            plt.scatter(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp), color='red')
            plt.text(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp),'({:.2f},{:.2f})'.format(*np.arange(0,5,0.01)[np.where(temp==np.max(temp))],np.max(temp)),horizontalalignment='left',verticalalignment='bottom')
            plt.ylabel(r'$\sigma (k)$')
            plt.tick_params('x', labelbottom=False)
            plt.legend()
        if i ==2:
            ax3 = plt.subplot(313, sharex=ax1,sharey=ax1)
            temp = (graphfinder(p,500,jac_0))
            plt.plot(np.arange(0,5,0.01),temp,label ='P= {:.2f}'.format(p))      
            plt.axhline(y=0.0, color='k', linestyle='--')
            plt.axis([0, 2, -0.5, 0.5])
            plt.scatter(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp), color='red')
            plt.text(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp),'({:.2f},{:.2f})'.format(*np.arange(0,5,0.01)[np.where(temp==np.max(temp))],np.max(temp)),horizontalalignment='left',verticalalignment='bottom')
            plt.xlabel('k')
            plt.tick_params('x', labelbottom=True)
            plt.legend()
    # =============================================================================
    #     if i==3 :
    #         ax4 = plt.subplot(314, sharex=ax1,sharey=ax1)
    #         temp = (graphfinder(p,500,jac_0))
    #         plt.plot(np.arange(0,5,0.01),temp,label ='P= {}'.format(p))           
    #         plt.axhline(y=0.0, color='k', linestyle='--')
    #         plt.axis([0, 2, -0.5, 0.5])
    #         plt.scatter(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp), color='red')
    #         plt.text(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp),'({:.2f},{:.2f})'.format(*np.arange(0,5,0.01)[np.where(temp==np.max(temp))],np.max(temp)),horizontalalignment='left',verticalalignment='bottom')
    #         plt.legend()
    # =============================================================================
    # =============================================================================
    #     if i==4 :
    #         ax5 = plt.subplot(315, sharex=ax1,sharey=ax1)
    #         temp = (graphfinder(p,500,jac_0))
    #         plt.plot(np.arange(0,5,0.01),temp,label ='P= {}'.format(p))         
    #         plt.axhline(y=0.0, color='k', linestyle='--')
    #         plt.axis([0, 2, -0.5, 0.5])
    #         plt.scatter(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp), color='red')
    #         plt.text(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp),'({:.2f},{:.2f})'.format(*np.arange(0,5,0.01)[np.where(temp==np.max(temp))],np.max(temp)),horizontalalignment='left',verticalalignment='bottom')
    #         plt.legend()    
    #     if i==5 :
    #         ax6 = plt.subplot(316, sharex=ax1,sharey=ax1)
    #         temp = (graphfinder(p,500,jac_0))
    #         plt.plot(np.arange(0,5,0.01),temp,label ='P= {}'.format(p))         
    #         plt.axhline(y=0.0, color='k', linestyle='--')
    #         plt.axis([0, 2, -0.5, 0.5])
    #         plt.scatter(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp), color='red')
    #         plt.text(np.arange(0,5,0.01)[np.where(temp==np.max(temp))], np.max(temp),'({:.2f},{:.2f})'.format(*np.arange(0,5,0.01)[np.where(temp==np.max(temp))],np.max(temp)),horizontalalignment='left',verticalalignment='bottom')
    #         plt.legend() 
    # =============================================================================
    # =============================================================================
     
    plt.show()
        
         
