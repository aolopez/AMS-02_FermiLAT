# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:49:33 2015

@author: Alejandro
"""

import numpy as np
import pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import spence
import math as m
import time
from mpmath import polylog
start=time.time()

alpha=1/137.
mu=0.10565837
def polylog(x):
    return spence(1-x)
def S(x):
    if x<1:
        return (2-x+2*x*m.log(x)-x**2)/x
    else:
        return 0.
def S1(x):
    if x<1:
        return (1+(1-x)**2)/x
    else:
        return 0.
alpha=1/137.
def integrand(E,mx):
    if S(E/mx)>0:
        return E*10**(-25)*10**(18)*(1/(0.001*mx)**3)*S(E/mx)*(2./1000.)*m.log(1./mu)*(alpha/(8.*m.pi*m.pi)) # need to multiply by 2?? - esta version ya tiene el 2
    else:
        return 0.
mp=100.
def integrandnew(E,mx):
    y=E/mx    
    if y<1:
        S1=2-y**2 + 2*y*m.log((mp-mu)*y/(mp-2*mu))-(mp**2 - 2*mu**2)*y/((mp-mu)*(mp - 2*mu))
        S2=2*mu**2*(2-m.log(mu**2*y**2/((mp-2*mu)**2*(1-y))))-3*mu*mp*(4./3.-m.log(mu*y**2*(mp-mu)/((mp-2*mu)**2*(1-y))))+mp**2*(1-m.log(y**2*(mp-mu)**2/((mp-2*mu)**2*(1-y))))
        const=E*10**(-25)*10**(18)*(1/(0.001*mx))**2*(1/(8.*m.pi))        
        return const*(2*alpha/(m.pi*y*mx))*(y**2 + 2*y*(polylog((mp-2*mu)/(mp-mu))-polylog(y))+(2-y**2)*m.log(1-y)+(m.log(mp**2/mu**2)-1)*S1-y*S2/(2*mu**2 - 3*mp*mu+mp**2)) 
    else:
        return 0.
def integrand1(E,mx):
    const=E*10**(-25)*10**(18)*(1/(0.001*mx))**2*(1/(8.*m.pi))
    def function(x):
        if x<1:
            return const*(alpha/m.pi)*((1+(1-x)**2)/x)*(m.log(mp**2*(1-x)/mu**2)-1)*(2./(x*mx))
        else:
            return 0.
    y=E/mx
#    const1=(mp-2*mu)/(mp-mu)
    return quad(function,y,1)[0]
    
    
def pred(emin,emax,mx):
    return quad(integrand1,emin,emax,args=(mx))[0]
J=[10**(18.8),10**(17.7),10**(17.9),10**(18.1),10**(19.),10**(18.8),10**(18.2),10**(18.1),10**(17.7),10**(17.6),10**(17.9),10**(18.6),10**(19.5),10**(18.4),10**(18.3),10**(19.3),10**(18.8),10**(19.1)]    
files=["like_bootes_I.txt","like_canes_venatici_I.txt","like_canes_venatici_II.txt","like_carina.txt","like_coma_berenices.txt","like_draco.txt","like_fornax.txt","like_hercules.txt","like_leo_I.txt","like_leo_II.txt","like_leo_IV.txt","like_sculptor.txt","like_segue_1.txt","like_sextans.txt","like_ursa_major_I.txt","like_ursa_major_II.txt","like_ursa_minor.txt","like_willman_1.txt"]
J1=[]
files1=[]
for i in np.arange(len(J)):
    if (i!=1) and (i!=8) and (i!=14):
        J1.append(J[i])
        files1.append(files[i])
J=J1
files=files1
def dLike(sig,index,mx):
    likefile = files[index]
    data = np.loadtxt(likefile, unpack=True)
    emins, emaxs = np.unique(data[0]),np.unique(data[1])
#    ebin = np.sqrt(emins*emaxs)
    efluxes = data[2].reshape(len(emins),-1)
    logLikes = data[3].reshape(len(emins),-1)
    
    pred1=np.array([pred(e1,e2,mx) for e1,e2 in zip(emins,emaxs)])
    likes = [ interp1d(f,l-l.max(),bounds_error=True,fill_value=0.) for f,l in zip(efluxes,logLikes) ]
    like = lambda c: sum([lnlfn(c*p) for lnlfn,p in zip(likes,pred1)])
    
    epsilon = 1e-4 # Just to make sure we stay within the interpolation range
    xmin =  epsilon
    xmax = np.log10(efluxes.max()/efluxes[efluxes>0].min()) - epsilon
    x = np.logspace(xmin,xmax,250)
    
    norm0 = efluxes[efluxes>0].min() / pred1.max()
    norms = norm0 * x
    j0=10**(18.)
    sigmav0=10**(-25.)
    lnl = np.array([like(n) for n in norms])
    jfactor=J[index]
    sigmav = j0/jfactor * sigmav0 * norms
    lnlfn = interp1d(sigmav,lnl,bounds_error=True,fill_value=0.)
    return lnlfn(sig),sigmav.min(),sigmav.max()
#    mle = lnl.max()
#    sigmav_mle = sigmav[lnl.argmax()] 
#    delta = 2.71/2
#    limit = brentq(lambda x: lnlfn(x)-mle+delta, sigmav_mle, sigmav.max(),
#                   xtol=1e-10*sigmav_mle)

#fLike=lambda x,mx: sum([dLike(x,i,mx)[0] for i in np.arange(len(files))])
def fLike(x,mx):
    maxsig=[]
    y=[]
    minsig=[]
    for i in np.arange(len(files)):
        A=dLike(x,i,mx)
        y.append(A[0])
        maxsig.append(A[1])
        minsig.append(A[2])
    return sum(y),max(maxsig),min(minsig)

def maxLike(mx):
    if mx<4e5:
        init=10**(-22)
    else:
        init=10**(-20)
    B=fLike(init,mx)
    maxsig=B[1]
    epsilon=1.1*maxsig
    delta=2.
    minsig=B[2]
    if fLike(epsilon,mx)[0]<fLike(maxsig,mx)[0] and fLike(epsilon*10.,mx)[0]<fLike(maxsig,mx)[0]:
        limit=brentq(lambda x: fLike(x,mx)[0]-fLike(maxsig,mx)[0]+delta,maxsig,minsig,xtol=1e-10*maxsig)
        return limit
    else:
        return "need to write code for finding max"
def plot(mxin,mxf,numbers,text):
    f1=open("results_"+text+".txt","w")
    mi=m.log10(mxin)
    mo=m.log10(mxf)
    x=np.logspace(mi,mo,numbers)
    y=[]
    for i in np.arange(len(x)):
        res=maxLike(x[i])
        y.append(res)
        f1.write(str(x[i])+" "+str(res)+"\n")
        print("iteration: "+str(len(x)-i)+" and time elapsed: "+ str((time.time()-start)/60.)+" mins")
    f1.close()
#    plt.loglog(x,y)