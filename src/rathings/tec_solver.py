from rathings.phase_unwrap import phase_unwrapp1d
import numpy as np
TECU = 1e12

def l1data_l2model_solve(obs_phase,freqs,Cd_error,Ct_ratio=0.01,m0=None, CS_solve=False):
    '''Solves for the terms phase(CS,TEC) = CS + e^2/(4pi ep0 me c) * TEC/nu
    
    Delay is taken out.
    If CS is optionally solved for with option `CS_solve`.

    `obs_phase` : `numpy.ndarray`
        The phase as observable in radians. The shape is assumed to be (len(freqs), num_timestamps)
    `freqs` : `numpy.ndarray`
        The frequencies in Hz at midpoints of observables.
    `Cd_error` : `float` or `numpy.ndarray`
        The uncertainty of the measured `obs_phase` in degrees.
        If not a float then must be of shape `obs_phase.shape`
    `Ct_ratio` : `float` (optional)
        The systematic uncertainty in fraction of absolute phase.
        Ct will be calculated as Ct_ratio*abs(obs_phase)
    `m0` : `numpy.ndarray` (optional)
        The initial guess of the model. If not given then set to zeros.
        Shape must be (num_timestamps, 2) [even if CS is not solved for]
        m0[:,0] = tec0, m0[:,1] = CS0
    `CS_solve` : bool (optional)
        Whether or not to solve for CS or set it to 0.
 
    Returns model of shape (num_timestamps, 2) where,
    m[:,0] = tec, m[:,1] = CS
    '''
    alpha = 0.
    if CS_solve:
        alpha = 1.
    #whethre or not to use a priori information (regularization) (soft problem makes little difference)
    beta = 0.
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
        m0 = np.zeros([obs_phase.shape[1],2],dtype=np.double)
    m = m0.copy()
        
    cov_m = np.array([1e-4,1e-4])
    #print( calc_phase(m,freqs) - obs_phase)
    
    
    Ct = (Ct_ratio*np.abs(obs_phase))**2
    Cd = (Cd_error*np.pi/180.)**2
    CdCt = Cd+Ct
    #print(np.sqrt(CdCt))
    #print( np.sum(np.abs(calc_phase(m,freqs) - obs_phase)/np.sqrt(CdCt),axis=0))
    S = neglogL(obs_phase,m,CdCt,m0,cov_m,freqs)
    #print("Initial neglogL: {}".format(S))
    iter = 0
    Nmax = 2#one is enough
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
    #print("Final neglogL: {}".format(S))
    return m
