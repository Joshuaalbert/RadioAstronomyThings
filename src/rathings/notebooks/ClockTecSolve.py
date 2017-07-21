
# coding: utf-8

# In[ ]:

#!/bin/env python

"""
This solves for the terms common scalar phase, tec, and complex gain using MH algorithm.
author: Joshua Albert
albert@strw.leidenuniv.nl
"""

import numpy as np
import pylab as plt
from dask import delayed

def phase_unwrapp1d(theta):
    '''the difference between two timesteps is unaliased by assumption so theta_i+1 - theta_i < pi.
    So Wrap(theta_i+1 - theta_i) is the real gradient and we can integrate them'''
    def wrap(phi):
        res = 1j*phi
        np.exp(res,out=res)
        return np.atleast_1d(np.angle(res))
    grad = wrap(theta[1:] - theta[:-1])
    unwrapped_theta = np.zeros(len(theta),dtype=np.double)
    np.cumsum(grad,out=unwrapped_theta[1:])
    unwrapped_theta += theta[0]
    return unwrapped_theta
    

TECU = 10e16

def vertex(x1,x2,x3,y1,y2,y3):
    '''Given three pairs of (x,y) points return the vertex of the
         parabola passing through the points. Vectorized and common expression reduced.'''
    #Define a sequence of sub expressions to reduce redundant flops
    x0 = 1/x2
    x4 = x1 - x2
    x5 = 1/x4
    x6 = x1**2
    x7 = 1/x6
    x8 = x2**2
    x9 = -x7*x8 + 1
    x10 = x0*x1*x5*x9
    x11 = 1/x1
    x12 = x3**2
    x13 = x11*x12
    x14 = 1/(x0*x13 - x0*x3 - x11*x3 + 1)
    x15 = x14*y3
    x16 = x10*x15
    x17 = x0*x5
    x18 = -x13 + x3
    x19 = y2*(x1*x17 + x14*x18*x6*x9/(x4**2*x8))
    x20 = x2*x5
    x21 = x11*x20
    x22 = x14*(-x12*x7 + x18*x21)
    x23 = y1*(-x10*x22 - x21)
    x24 = x16/2 - x19/2 - x23/2
    x25 = -x17*x9 + x7
    x26 = x0*x1*x14*x18*x5
    x27 = 1/(-x15*x25 + y1*(x20*x7 - x22*x25 + x7) + y2*(-x17 + x25*x26))
    x28 = x24*x27
    return x28,x15 + x22*y1 + x24**2*x27 - x26*y2 + x28*(-x16 + x19 + x23)




    
def clock_tec_solveMH(obs_phase, freqs, times, m0, cov, Cd_error, Ct_ratio, plot = False):
    '''Solves for the terms phase(CS,TEC,delay) = CS + e^2/(4pi ep0 me c) * TEC/nu + 2pi*nu*delay
    
    Assumes phase is in units of radians, freqs in is units of Hz, 
    and times is in units of seconds with arbitrary offset
    
    obs_phase is shape (num_freqs, num_times)'''
     
    binning = 50
    convergence = binning**2 * 3
    def calc_phase(m, freqs):
        phase = np.multiply.outer(np.ones(len(freqs)),m[:,2]) + 8.44797256e-7*TECU * np.multiply.outer(1./freqs,m[:,1]) + 2.*np.pi*np.multiply.outer(freqs,m[:,0])
        return phase
    
    def neglogL(obs_phase,phase,CdCt):
        L2 = obs_phase - phase
        L2 *= L2
        L2 /= (CdCt+1e-15)
        return np.sum(L2,axis=0)/2.
    
    def sample_prior(last, cov):
        """Last is tau,tec,cs in matrix of size [len(times),3], return similar shaped next point"""
        return last + np.random.multivariate_normal(mean = [0,0,0], cov=cov,size = last.shape[0])
    cs = m0[:,2]
    tec = m0[:,1]
    tau = m0[:,0]
    print("Initial CS: {}".format(cs))
    print("Initial TEC: {}".format(tec))
    print("Initial delay: {}".format(tau))
    m = m0.copy()
#     if plot:
#         plt.plot(times,cs0,label="CS0")
#         plt.plot(times,tec0,label="TEC0")
#         plt.plot(times,delay0,label="delay0")
#         plt.legend(frameon=False)
#         plt.show()
    Ct = (Ct_ratio*np.abs(obs_phase))**2
    Cd = (Cd_error*np.pi/180.)**2
    CdCt = Cd+Ct
    Si = neglogL(obs_phase,calc_phase(m,freqs),CdCt)
    print("Initial Si: {}".format(Si))
    max_iter = 100*convergence
    posterior = np.zeros([convergence,len(times),3],dtype=np.double)
    multiplicity = np.zeros([convergence,len(times)],dtype=np.double)
    posterior[0,:,:] = m
    minS = Si
    minSol = m.copy()
    accepted = np.ones(len(times),dtype=np.int)
    cov_prior = np.diag([1e-10, 1e-6,0.5])**2 + cov
    iter = 1
    while np.max(accepted) < convergence and iter < max_iter:
        #sample
        last = np.array([posterior[accepted[i] - 1,i,:] for i in range(len(times))])
        m_j = sample_prior(last,cov_prior)
        
        Sj = neglogL(obs_phase,calc_phase(m_j,freqs),CdCt)
        Lj = np.exp(-Sj)
        
        accept_mask = np.bitwise_or(Sj < Si, np.log(np.random.uniform(size=len(Sj))) < Si - Sj)
        #print(accept_mask)
        Si[accept_mask] = Sj[accept_mask]
        for i in range(len(times)):
            if accept_mask[i]:
                posterior[accepted[i],i,:] = m_j[i,:]
                multiplicity[accepted[i],i] += 1
                accepted[i] += 1
            else:
                multiplicity[accepted[i]-1,i] += 1
                
        if np.any(accept_mask):
            #print(m_j)
            #print("{} accepted".format(np.sum(accept_mask)))
            pass
                
        maxL_mask = Sj < minS
        minSol[maxL_mask,:] = m_j[maxL_mask]
        minS[maxL_mask] = Sj[maxL_mask]
        iter += 1
    if iter != max_iter:
        print("Converged in {} steps with mean acceptance rate of {}".format(iter,np.mean(accepted)/iter))
    posteriors = []
    multiplicities = []
    means = []
    stds = []
    maxLs = []
    for i in range(len(times)):
        posteriors.append(posterior[:accepted[i],i,:])
        multiplicities.append(multiplicity[:accepted[i],i])
        means.append(np.sum(posteriors[i].T*multiplicities[i],axis=1)/np.sum(multiplicities[i]))
        stds.append(np.sqrt(np.sum(posteriors[i].T**2*multiplicities[i],axis=1)/np.sum(multiplicities[i]) - means[i]**2))
        maxLs.append(minSol[i,:])
        print ("Sol {}, (Gaussian) sol is {} +- {}".format(i, means[i],stds[i]))
        print("    maxL sol is {}".format(maxLs[i]))
    if plot:
        plt.hist(posteriors[0][:,0],weights = multiplicities[0],label='tau')
        plt.legend(frameon=False)
        plt.show()
        plt.hist(posteriors[0][:,1],weights = multiplicities[0],label='tec')
        plt.legend(frameon=False)
        plt.show()
        plt.hist(posteriors[0][:,2],weights = multiplicities[0],label='cs')
        plt.legend(frameon=False)
        plt.show()
    return maxLs

@delayed
def clock_tec_solve_dask(obs_phase, freqs, m0, cov, Cd_error, Ct_ratio, plot=False):
    '''Solves for the terms phase(CS,TEC,delay) = CS + e^2/(4pi ep0 me c) * TEC/nu + 2pi*nu*delay
    
    Assumes phase is in units of radians, freqs in is units of Hz
    
    obs_phase is shape (num_freqs)'''
     
    binning = 50
    convergence = binning**2 * 3
    def calc_phase(m, freqs):
        tau,tec,cs = m[:]
        phase = freqs*(tau * np.pi * 2.)
        phase += (8.44797256e-7 * TECU * tec)/freqs
        phase += cs
        return phase
    
    def neglogL(obs_phase,phase,CdCt):
        L2 = obs_phase - phase
        L2 *= L2
        L2 /= (CdCt+1e-15)
        return np.sum(L2)/2.
    
    def sample_prior(last, cov):
        """Last is tau,tec,cs in matrix of size [len(times),3], return similar shaped next point"""
        return last + np.random.multivariate_normal(mean = [0,0,0], cov=cov)
        #m0 = last[0] + np.random.uniform(low = -np.sqrt(cov[0,0])/2., high = np.sqrt(cov[0,0])/2.)
        #m1 = last[1] + np.random.uniform(low = -np.sqrt(cov[1,1])/2., high = np.sqrt(cov[1,1])/2.)
        #m2 = last[2] + np.random.uniform(low = -np.sqrt(cov[2,2])/2., high = np.sqrt(cov[2,2])/2.)
        #return np.array([m0,m1,m2])
    cs = m0[2]
    tec = m0[1]
    tau = m0[0]
    print("Initial CS: {}".format(cs))
    print("Initial TEC: {}".format(tec))
    print("Initial delay: {}".format(tau))
    m = m0.copy()

    Ct = (Ct_ratio*np.abs(obs_phase))**2
    Cd = (Cd_error*np.pi/180.)**2
    CdCt = Cd+Ct
    Si = neglogL(obs_phase,calc_phase(m,freqs),CdCt)
    print("Initial Si: {}".format(Si))
    max_iter = 100*convergence
    posterior = np.zeros([convergence,3],dtype=np.double)
    multiplicity = np.zeros([convergence],dtype=np.double)
    posterior[0,:] = m
    minS = Si
    minSol = m.copy()
    accepted = 1
    cov_prior = np.diag([1e-9, 1e-4,1e-1])**2 + cov
    iter = 1
    while accepted < convergence and iter < max_iter:
        #sample
        #m_j = sample_prior(posterior[accepted-1],cov_prior)
        m_j = sample_prior(minSol,cov_prior)
        #m_j = sample_prior(m0,cov_prior)
        Sj = neglogL(obs_phase,calc_phase(m_j,freqs),CdCt)
        Lj = np.exp(-Sj)
        if Sj < Si or np.log(np.random.uniform()) < Si - Sj:
            Si = Sj
            posterior[accepted,:] = m_j
            multiplicity[accepted] += 1
            accepted += 1
        else:
            multiplicity[accepted-1] += 1
        if Sj < minS:
            minSol = m_j
            minS = Sj
        iter += 1
    if iter != max_iter:
        print("Converged in {} steps with mean acceptance rate of {}".format(iter,np.mean(accepted)/iter))
    posterior = posterior[:accepted,:]
    multiplicity = multiplicity[:accepted,]
    mean = np.sum(posterior.T*multiplicity,axis=1)/np.sum(multiplicity)
    std = np.sqrt(np.sum(posterior.T**2*multiplicity,axis=1)/np.sum(multiplicity) - mean**2)
    maxLsol = minSol
    print ("(Gaussian) sol is {} +- {}".format(mean,std))
    print("    maxL sol is {}".format(maxLsol))
    if plot:
        plt.hist(posteriors[0][:,0],weights = multiplicities[0],label='tau')
        plt.legend(frameon=False)
        plt.show()
        plt.hist(posteriors[0][:,1],weights = multiplicities[0],label='tec')
        plt.legend(frameon=False)
        plt.show()
        plt.hist(posteriors[0][:,2],weights = multiplicities[0],label='cs')
        plt.legend(frameon=False)
        plt.show()
    return maxLsol
        
def clock_test_solve_both(obs_phase, freqs, times, m0, cov, Cd_error, Ct_ratio):
    m,cov = least_squares_solve(obs_phase, freqs, times,Cd_error,Ct_ratio)
    m = clock_tec_solveMH(phase, freqs, times, m, np.mean(cov,axis=0), Cd_error, Ct_ratio,plot=True)
    return m      
                          
            
def test_clock_tec_solveMH():
    times = np.arange(2)
    freqs = np.linspace(110e6,170e6,100)
    cs = times*0.01
    tec = np.random.uniform(size=len(times))*0.01
    delay = np.ones(len(times)) * 1e-9# 10ns
    phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,tec) + 2.*np.pi*np.multiply.outer(freqs,delay)
    phase += 10.*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    plt.imshow(phase,origin='lower',extent=(times[0],times[-1],freqs[0],freqs[-1]),aspect='auto')
    plt.colorbar()
    plt.xlabel('times (s)')
    plt.ylabel('freqs (Hz)')
    plt.show()
    clock_tec_solveMH(phase, freqs, times, plot=True)
    
def test_clock_tec_solve_l1l2():
    times = np.arange(2)
    freqs = np.linspace(110e6,170e6,100)
    cs = times*0.00001
    tec = np.random.uniform(size=len(times))*0.01
    phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,tec)
    phase += 10.*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    plt.imshow(phase,origin='lower',extent=(times[0],times[-1],freqs[0],freqs[-1]),aspect='auto')
    plt.colorbar()
    plt.xlabel('times (s)')
    plt.ylabel('freqs (Hz)')
    plt.show()
    m0,cov = least_squares_solve(phase, freqs, times,10,Ct_ratio=0.01)
    print(m0)
    l1data_l2model_solve(phase,freqs,times,10,Ct_ratio=0.01,m0=m0)
    
def test_clock_tec_solve_error():
    import pylab as plt
    times = np.arange(100)
    Cd_errors = np.linspace(1,100,100)
   
    f,(ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    for num_freq in [10,100,1000,10000]:
        freqs = np.linspace(110e6,170e6,num_freq)
        sol_acc = []
        for Cd_error in Cd_errors:
            cs = times
            tec = np.random.uniform(size=len(times))*0.01
            delay = np.ones(len(times)) * 1e-9# 10ns
            phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,tec) + 2.*np.pi*np.multiply.outer(freqs,delay)
            phase += Cd_error*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
            #plt.imshow(phase,origin='lower',extent=(times[0],times[-1],freqs[0],freqs[-1]),aspect='auto')
            #plt.colorbar()
            #plt.xlabel('times (s)')
            #plt.ylabel('freqs (Hz)')
            #plt.show()
            m,cov = least_squares_solve(phase, freqs, times,Cd_error,Ct_ratio=0.01)
            m_exact = np.array([delay,tec,cs]).T
            sol_acc.append(np.mean(np.abs(m - m_exact),axis=0))
        sol_acc_ = np.array(sol_acc)
        ax1.plot(Cd_errors,sol_acc_[:,0])
        #plt.show()
        ax2.plot(Cd_errors,sol_acc_[:,1])
        #plt.show()
        ax3.plot(Cd_errors,sol_acc_[:,2])
    plt.show()
                          
def test_clock_tec_solve():
    import pylab as plt
    times = np.arange(2)
    freqs = np.linspace(110e6,170e6,1000)
    
    cs = np.array([1,1])*0
    tec = np.array([0.1,0.2])
    delay = np.ones(len(times)) * 2e-9# 10ns
    phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,tec)# + 2.*np.pi*np.multiply.outer(freqs,delay)
    phase += 5*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    #plt.imshow(phase,origin='lower',extent=(times[0],times[-1],freqs[0],freqs[-1]),aspect='auto')
    #plt.colorbar()
    #plt.xlabel('times (s)')
    #plt.ylabel('freqs (Hz)')
    #plt.show()
    m,cov = least_squares_solve(phase, freqs, times,5,Ct_ratio=0.01)
    print(m)
    #m_exact = np.array([delay,tec,cs]).T
    #clock_tec_solveMH(phase, freqs, times, m, np.max(cov,axis=0), 5, 0.01, plot = True)
    
def test_clock_tec_solve_dask():
    np.random.seed(1234)
    import pylab as plt
    times = np.arange(2)
    freqs = np.linspace(110e6,170e6,1000)
    
    cs = np.array([1,1])
    tec = np.array([0.1,0.2])
    delay = np.ones(len(times)) * 2e-9# 10ns
    phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,tec) + 2.*np.pi*np.multiply.outer(freqs,delay)
    phase += 15*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    #plt.imshow(phase,origin='lower',extent=(times[0],times[-1],freqs[0],freqs[-1]),aspect='auto')
    #plt.colorbar()
    #plt.xlabel('times (s)')
    #plt.ylabel('freqs (Hz)')
    #plt.show()
    m,cov = least_squares_solve(phase, freqs, times,15,Ct_ratio=0.01)
    m_exact = np.array([delay,tec,cs]).T
    import dask.array as da
    solsMH = [da.from_delayed(clock_tec_solve_dask(phase[:,i],freqs,m[i,:], cov[i,:,:],15,0.01),shape=(3,),dtype=np.double) for i in range(len(times))]
    
    sol_stacked = da.stack(solsMH, axis = 0)
    sol = sol_stacked.compute()
    print(sol)
        
if __name__ == '__main__':
    #test_clock_tec_solve()
    #a b ctest_clock_tec_solve()
    #test_clock_tec_solve_dask()
    test_clock_tec_solve_l1l2()


# In[ ]:

def derive_solution():
    from sympy import symbols, Matrix, cse, cos, sin, Abs, Rational,acos,asin

    cs,K,tec,nu,phase,sigma_phase,alpha,beta,tec_p,cs_p,sigma_tec,sigma_cs = symbols('cs K tec nu phase sigma_phase alpha beta tec_p cs_p sigma_tec sigma_cs', real=True)
    g = K*tec/nu + cs*alpha

    L = Abs(g - phase)/sigma_phase + beta*((tec - tec_p)**Rational(2)/sigma_tec**Rational(2)/Rational(2) + (cs -  cs_p)**Rational(2)/sigma_cs**Rational(2)/Rational(2))

    req,res = cse(L,optimizations='basic')
    for line in req:
        print("{} = {}".format(line[0],line[1]).replace("Abs","np.abs").replace("cos","np.cos").replace("sin","np.sin").replace("sign","np.sign"))
    print("{}".format(res[0]).replace("Abs","np.abs").replace("cos","np.cos").replace("sin","np.sin").replace("sign","np.sign"))
    print()

    grad = Matrix([sigma_tec**Rational(2)*L.diff(tec), sigma_cs**Rational(2)*L.diff(cs)])
    req,res = cse(grad,optimizations='basic')
    for line in req:
        print("{} = {}".format(line[0],line[1]).replace("Abs","np.abs").replace("cos","np.cos").replace("sin","np.sin").replace("sign","np.sign"))
    print("{}".format(res[0]).replace("Abs","np.abs").replace("cos","np.cos").replace("sin","np.sin").replace("sign","np.sign"))
    print()

    H = Matrix([[L.diff(tec).diff(tec),L.diff(tec).diff(cs)],[L.diff(cs).diff(tec),L.diff(cs).diff(cs)]])

    req,res = cse(H,optimizations='basic')
    for line in req:
        print("{} = {}".format(line[0],line[1]).replace("Abs","np.abs").replace("cos","np.cos").replace("sin","np.sin").replace("sign","np.sign"))
    print("{}".format(res[0]).replace("Abs","np.abs").replace("cos","np.cos").replace("sin","np.sin").replace("sign","np.sign"))
derive_solution()


# In[ ]:


import numpy as np
import pylab as plt

def phase_unwrapp1d(theta):
    '''the difference between two timesteps is unaliased by assumption so theta_i+1 - theta_i < pi.
    So Wrap(theta_i+1 - theta_i) is the real gradient and we can integrate them'''
    def wrap(phi):
        res = 1j*phi
        np.exp(res,out=res)
        return np.atleast_1d(np.angle(res))
    grad = wrap(theta[1:] - theta[:-1])
    unwrapped_theta = np.zeros(len(theta),dtype=np.double)
    np.cumsum(grad,out=unwrapped_theta[1:])
    unwrapped_theta += theta[0]
    return unwrapped_theta
    

TECU = 10e16

def least_squares_solve(obs_phase,freqs,times,Cd_error,Ct_ratio=0.01,m0=None):
    '''Solves for the terms phase(CS,TEC,delay) = CS + e^2/(4pi ep0 me c) * TEC/nu + 2pi*nu*delay
    
    Assumes phase is in units of radians, freqs in is units of Hz, 
    and times is in units of seconds with arbitrary offset
    
    obs_phase is shape (num_freqs, num_times)'''
    
    f = np.multiply.outer(freqs,np.ones(len(times)))
    
    def calc_phase(cs,tec,delay, freqs):
        phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU * np.multiply.outer(1./freqs,tec) + 2.*np.pi*np.multiply.outer(freqs,delay)
        return phase
    
    def neglogL(obs_phase,phase,CdCt):
        '''Return per timestep'''
        L2 = obs_phase - phase
        L2 *= L2
        L2 /= (CdCt + 1e-15)
        return np.sum(L2,axis=0)/2.
    
    def calc_grad(cs,tec,delay,f,obs_phase,CdCt):
        grad = np.zeros([len(times),3],dtype=np.double)
        phase = calc_phase(cs,tec,delay,f[:,0])
        dd = obs_phase - phase
        dd /= CdCt
        #tau comp
        gtau = dd*f
        gtau *= -2.*np.pi
        gtau = np.sum(gtau,axis=0)
        #tec comp
        gtec = dd/f
        gtec *= -8.44797256e-7*TECU
        gtec = np.sum(gtec,axis=0)
        #cs comp
        gcs = -dd
        gcs = np.sum(gcs,axis=0)
        grad[:,0] = gtau
        grad[:,1] = gtec
        grad[:,2] = gcs
        return grad
    
    def calc_Hessian(f,CdCt):
        H = np.zeros([len(times),3,3],dtype = np.double)
        x0 = f/CdCt
        H[:,0,0] = np.sum(4*np.pi**2 * x0*f,axis=0)
        H[:,0,1] = np.sum(2*np.pi*8.44797256e-7*TECU/CdCt,axis=0)
        H[:,0,2] = np.sum(2*np.pi*x0,axis=0)
        H[:,1,1] = np.sum((8.44797256e-7*TECU)**2/(f**2*CdCt),axis=0)
        H[:,1,2] = np.sum(8.44797256e-7*TECU/(f*CdCt),axis=0)
        H[:,2,2] = np.sum(1./CdCt,axis=0)
        H[:,1,0] = H[:,0,1]
        H[:,2,0] = H[:,0,2]
        H[:,2,1] = H[:,1,2]
        return H
    
    def inv_Hessian(H):
        a = H[:,0,0]
        b = H[:,0,1]
        c = H[:,0,2]
        d = H[:,1,1]
        e = H[:,1,2]
        f = H[:,2,2]
        x0 = 1/a
        x1 = b**2
        x2 = x0*x1
        x3 = 1/(d - x2)
        x4 = b*c
        x5 = e - x0*x4
        x6 = x3*x5
        x7 = b*x6 - c
        x8 = a*d
        x9 = -x1 + x8
        x10 = 1/(a*e**2 + c**2*d - 2*e*x4 + f*x1 - f*x8)
        x11 = x0*x10*x9
        x12 = x10*x9
        x13 = x0*x3*(-b + x12*x5*x7)
        x14 = -x11*x7
        x15 = x12*x6
        Hinv = np.zeros([len(a),3,3])
        Hinv[:,0,0] = x0*(-x11*x7**2 + x2*x3 + 1)
        Hinv[:,0,1] = x13
        Hinv[:,0,2] = x14
        Hinv[:,1,0] = Hinv[:,0,1]
        Hinv[:,1,1] = x3*(-x12*x3*x5**2 + 1)
        Hinv[:,1,2] = x15
        Hinv[:,2,0] = Hinv[:,0,2]
        Hinv[:,2,1] = Hinv[:,1,2]
        Hinv[:,2,2] = -x12
        return Hinv      
    
    def calc_epsilon_n(dir,tau_i,tec_i,cs_i,freqs,CdCt,obs_phase,step=1e-3):
        """Approximate stepsize"""
        g0 = calc_phase(cs_i,tec_i,tau_i, freqs)
        gp = calc_phase(cs_i+step*dir[:,2], tec_i + step*dir[:,1], tau_i + step*dir[:,0], freqs)
        Gm = (gp - g0)/step
        dd = obs_phase - g0
        epsilon_n = (np.sum(Gm*dd/CdCt,axis=0)/np.sum(Gm/CdCt*Gm,axis=0))
        return epsilon_n        
    
    if m0 is None:
        # come up with initial guess
        cs0  = np.zeros(len(times),dtype=np.double)
        delay0 = np.zeros(len(times),dtype=np.double)
        tec0 = np.zeros(len(times),dtype=np.double)
    #     # d/dnu (phase*nu) = cs + 4pi*nu*delay
        x0 = (freqs*obs_phase.T).T
        x1 = ((x0[1:,:] - x0[:-1,:]).T/(freqs[1:] - freqs[:-1])).T
    #     # d^2/dnu^2 (phase*nu) = 4pi*delay
        x2 = ((x1[1:,:] - x1[:-1,:]).T/(freqs[1:-1] - freqs[:-2])).T
        tau0 = np.mean(x2,axis=0)/4./np.pi
        x3 = 2*np.pi*np.multiply.outer(freqs,delay0)
        cs0 = np.mean(x1 - 2.*x3[1:,:],axis=0)
        cs = cs0*0
        tau = tau0
        tec = tec0
    else:
        cs = m0[:,2]
        tec = m0[:,1]
        tau = m0[:,0]
    #print("Initial CS: {}".format(cs0))
    #print("Initial TEC: {}".format(tec0))
    #print("Initial delay: {}".format(delay0))
    
    Ct = (Ct_ratio*np.abs(obs_phase))**2
    Cd = (Cd_error*np.pi/180.)**2
    CdCt = Cd+Ct
    S = neglogL(obs_phase,calc_phase(cs,tec,tau,freqs),CdCt)
    #print("Initial neglogL: {}".format(S))
    iter = 0
    Nmax = 1
    while iter < Nmax:
        grad = calc_grad(cs,tec,tau,f,obs_phase,CdCt)
        H = calc_Hessian(f,CdCt)
        Hinv = inv_Hessian(H)
        dir = np.einsum("ijk,ik->ij",Hinv,grad)
        epsilon_n = calc_epsilon_n(dir,tau,tec,cs,freqs,CdCt,obs_phase,step=1e-3)
        #print("epsilon_n: {}".format(epsilon_n))
        cs, tec, tau = cs+epsilon_n*dir[:,2], tec + epsilon_n*dir[:,1], tau + epsilon_n*dir[:,0]
        S = neglogL(obs_phase,calc_phase(cs,tec,tau,freqs),CdCt)
        m = np.array([tau,tec,cs]).T
        #print("Model: {}".format(m))
        #print("iter {}: neglogL: {}, log|dm/m|: {}, |grad|: {}".format(iter, S, np.mean(np.log(np.abs(np.einsum("i,ij->ij",epsilon_n,dir)/m))),np.sum(np.abs(grad))))
        iter += 1
    #print(Hinv)
    print("Final neglogL: {}".format(S))
    return m,Hinv

def l1data_l2model_solve(obs_phase,freqs,times,Cd_error,Ct_ratio=0.01,m0=None):
    '''Solves for the terms phase(CS,TEC,delay) = CS + e^2/(4pi ep0 me c) * TEC/nu
    
    Delay is taken out.
    
    Assumes phase is in units of radians, freqs in is units of Hz, 
    and times is in units of seconds with arbitrary offset
    
    obs_phase is shape (num_freqs, num_times)'''

    alpha = 1.
    beta = 1.
    def calc_phase(m, freqs):
        tec = m[:,0]
        cs = m[:,1]
        phase = 8.44797256e-7*TECU * np.multiply.outer(1./freqs,tec) + alpha*cs
        return phase
    
    def neglogL(obs_phase,m,CdCt_phase,m0,cov_m,freqs):
        '''Return per timestep'''
        K = 8.44797256e-7*TECU
        nu = np.multiply.outer(1./freqs,np.ones(obs_phase.shape[1]))
        tec = m[:,0]
        cs = m[:,1]
        tec_p = m0[:,0]
        cs_p = m0[:,1]
        sigma_tec2 = cov_m[0]
        sigma_cs2 = cov_m[1]
        sigma_phase = np.sqrt(CdCt_phase)
        phase = obs_phase
        #return np.sum(np.abs(K*np.multiply.outer(1./freqs,tec) - phase)/sigma_phase,axis=0)
    
        return beta*((tec - tec_p)**2/sigma_tec2 + (cs - cs_p)**2/sigma_cs2)/2 + np.sum(np.abs(K*np.multiply.outer(1./freqs,tec) + alpha*cs - phase)/sigma_phase,axis=0)
    
    def calc_grad(obs_phase,m,CdCt_phase,m0,cov_m,freqs):
        
        K = 8.44797256e-7*TECU
        nu = np.multiply.outer(1./freqs,np.ones(obs_phase.shape[1]))
        tec = m[:,0]
        cs = m[:,1]
        tec_p = m0[:,0]
        cs_p = m0[:,1]
        sigma_tec2 = cov_m[0]
        sigma_cs2 = cov_m[1]
        sigma_phase = np.sqrt(CdCt_phase)
        phase = obs_phase
        
        x0 = sigma_tec2
        x1 = K/nu
        x1_ = K*np.multiply.outer(1./freqs,tec)
        x2 = np.sign(alpha*cs - phase + x1_)/sigma_phase
        x3 = sigma_cs2
        
        grad = np.zeros([obs_phase.shape[1],2])
        grad[:,0] = x0*(beta*(tec - tec_p)/x0 + np.sum((x1*x2),axis=0))
        grad[:,1] = x3 * (beta*(cs - cs_p)/x3 + np.sum(alpha*x2,axis=0))
        return grad
    
    def calc_epsilon_n(dir,m,freqs,CdCt,obs_phase,step=1e-3):
        """Approximate stepsize"""
        g0 = calc_phase(m, freqs)
        gp = calc_phase(m + step*dir, freqs)
        Gm = (gp - g0)/step
        dd = obs_phase - g0
        epsilon_n = (np.sum(Gm*dd/CdCt,axis=0)/np.sum(Gm/CdCt*Gm,axis=0))
        return epsilon_n        
    
    if m0 is None:
        m = np.zeros([len(times),2],dtype=np.double)
    else:
        m = m0.copy()
        
    cov_m = np.array([1e-3,1e-3])
    #print( calc_phase(m,freqs) - obs_phase)
    
    
    Ct = (Ct_ratio*np.abs(obs_phase))**2
    Cd = (Cd_error*np.pi/180.)**2
    CdCt = Cd+Ct
    #print(np.sqrt(CdCt))
    #print( np.sum(np.abs(calc_phase(m,freqs) - obs_phase)/np.sqrt(CdCt),axis=0))
    S = neglogL(obs_phase,m,CdCt,m0,cov_m,freqs)
    print("Initial neglogL: {}".format(S))
    iter = 0
    Nmax = 3
    while iter < Nmax:
        grad = calc_grad(obs_phase,m,CdCt,m0,cov_m,freqs)
        dir = grad
        epsilon_n = calc_epsilon_n(dir,m,freqs,CdCt,obs_phase,step=1e-3)
        #print("epsilon_n: {}".format(epsilon_n))
        m += dir*epsilon_n
        S = neglogL(obs_phase,m,CdCt,m0,cov_m,freqs)
        #print("Model: {}".format(m))
        #print("iter {}: neglogL: {}, log|dm/m|: {}, |grad|: {}".format(iter, S, np.mean(np.log(np.abs(np.einsum("i,ij->ij",epsilon_n,dir)/m))),np.sum(np.abs(grad))))
        iter += 1
    #print(Hinv)
    print("Final neglogL: {}".format(S))
    return m

def test_clock_tec_solve_l1l2():
    np.random.seed(1234)
    times = np.arange(1)
    freqs = np.linspace(110e6,170e6,100)
    cs = times*0.00
    tec = times + 0.001
    phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,tec)
    phase += 10*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    #phase = np.angle(np.exp(1j*phase))
    m0 = np.array([[0.,0.1]])
    #m0,cov = least_squares_solve(phase, freqs, times,10,Ct_ratio=1)
    #print(m0)
    m = m0[:,1:3]

    m = l1data_l2model_solve(phase,freqs,times,10,Ct_ratio=1,m0=m0)
    rec_phase = np.multiply.outer(np.ones(len(freqs)),m[:,1]) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,m[:,0])
    print(m)
    plt.plot(freqs,phase)
    plt.plot(freqs,rec_phase)
    plt.show()
    
        
def test_clock_tec_solve_l1l2_tolerance():
    times = np.arange(1)
    freqs = np.linspace(110e6,170e6,100)
    cs = times+0
    tec0 = 0.0001
    tec = times+tec0
    phase = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*TECU*np.multiply.outer(1./freqs,tec)
    phase += 40*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    #phase = np.angle(np.exp(1j*phase))
    m0,cov = least_squares_solve(phase, freqs, times,10,Ct_ratio=1)
    dist = []
    final = []
    m0 = m0[:,1:3]
    for tec in np.linspace(tec0-5,tec0 + 5,1000):
        dist.append(tec - tec0)
        m0[:,0] = tec
        m0[:,1] = 0.
        m = l1data_l2model_solve(phase,freqs,times,40,Ct_ratio=0.01,m0=m0)
        #print(m)
        final.append(m[0,0] - tec0)
    plt.plot(dist,final)
    plt.show()
    
test_clock_tec_solve_l1l2()


# In[ ]:




# In[ ]:



