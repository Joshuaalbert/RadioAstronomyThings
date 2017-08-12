from rathings.phase_unwrap import phase_unwrapp1d 
import numpy as np
from scipy.optimize import least_squares

TECU = 1e16


def calc_phase(tec, freqs, cs = 0.):
    '''Return the phase (num_freqs, num_times) from tec and CS.
    `tec` : `numpy.ndarray`
        tec in TECU of shape (num_times,)
    `freqs` : `numpy.ndarray`
        freqs in Hz of shape (num_freqs,)
    `cs` : `numpy.ndarray` or float (optional)
        Can be zero to disregard (default)
    '''
    TECU=1e16
    phase = 8.44797256e-7*TECU * np.multiply.outer(1./freqs,tec) + cs  
    return phase

def robust_l2(obs_phase, freqs, solve_cs=True):
    '''Solve the tec and cs for multiple datasets.
    `obs_phase` : `numpy.ndarray`
        the measured phase with shape (num_freqs, )
    `freqs` : `numpy.ndarray`
        the frequencies at the datapoints (num_freqs,)
    `solve_cs` : (optional) bool
        Whether to solve cs (True)
    '''
    if solve_cs:
        def residuals(m, freqs, obs_phase):
            tec,cs = m[0],m[1]
            return calc_phase(tec,freqs,cs=cs) - obs_phase
    else:
        def residuals(m, freqs, obs_phase):
            tec,cs = m[0],m[1]
            return calc_phase(tec,freqs,cs=0.) - obs_phase
    nan_mask = np.bitwise_not(np.isnan(obs_phase))
    obs_phase_ = obs_phase[nan_mask]
    freqs_ = freqs[nan_mask]
    m0 = [0.0, 0.]
    m = least_squares(residuals,m0,loss='soft_l1',f_scale=90.*np.pi/180.,args=(freqs_,obs_phase_))
    if solve_cs:
        return m.x[0], m.x[1]
    else:
        return m.x[0], 0.
    
def robust_l2_parallel(obs_phase, freqs, solve_cs=True, num_threads = None):
    '''Solve the tec and cs for multiple datasets.
    `obs_phase` : `numpy.ndarray`
        the measured phase with shape (num_freqs, num_datasets)
    `freqs` : `numpy.ndarray`
        the frequencies at the datapoints (num_freqs,)
    `solve_cs` : (optional) bool
        Whether to solve cs (True)
    `num_threads` : (optional) `int`
        number of parallel threads to run. default None is num_cpu
    '''
    from dask import delayed, compute
    from dask.threaded import get
    from functools import partial
    dsk = {}
    N = obs_phase.shape[1]
    values = [delayed(partial(robust_l2, solve_cs=solve_cs), pure=True)( obs_phase[:,i], freqs) for i in range(N)]
    results = compute(*values, get=get, num_workers=num_threads)
    return results

def l1_lpsolver(obs_phase, freqs, sigma_max = np.pi, fout=0.5, solve_cs=True, problem_name="l1_tec_solver"):
    '''Formulate the linear problem:
    Minimize 1'.(z1 + z2) s.t.
        phase = (z1 - z2) + K/nu*TEC
        |z1 + z2| < max_allowed
        min_tec < TEC < max_tec
    assumes obs_phase and freqs are for a single timestamp.
    '''
    nan_mask = np.isnan(obs_phase)
    obs_phase = obs_phase[np.bitwise_not(nan_mask)]
    freqs = freqs[np.bitwise_not(nan_mask)]
    if (len(freqs)<10):
        return 0.,0.,0.
    obs_phase_unwrap = phase_unwrapp1d(obs_phase)
    K = 8.44797256e-7*TECU
    #z+, z-, a+, a-, asigma, a, sigma, tec, cs
    N = len(freqs)
    ncols = N*6 + 3
    A_eq, b_eq = [],[]
    A_lt, b_lt = [],[]
    A_gt, b_gt = [],[]
    c_obj = np.zeros(ncols,dtype=np.double)
    for i in range(N):
        idx_p = i
        idx_m = N + i
        idx_ap = 2*N + i
        idx_am = 3*N + i
        idx_as = 4*N + i
        idx_a = 5*N + i
        idx_s = 6*N
        idx_tec = 6*N + 1
        idx_cs = 6*N + 2
        # 0<= a+ <= asigma
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_ap,idx_as]] = 1., -1.
        A_lt.append(row)
        b_lt.append(0.)
        # 0 <= z+ - a+ <= sigma - asigma
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_p,idx_ap, idx_s, idx_as]] = 1., -1., -1., 1.
        A_lt.append(row)
        b_lt.append(0.)
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_p,idx_ap]] = 1., -1.
        A_gt.append(row)
        b_gt.append(0.)
        #same for a-
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_am,idx_as]] = 1., -1.
        A_lt.append(row)
        b_lt.append(0.)
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_m,idx_am, idx_s, idx_as]] = 1., -1., -1., 1.
        A_lt.append(row)
        b_lt.append(0.)
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_m,idx_am]] = 1., -1.
        A_gt.append(row)
        b_gt.append(0.)
        # 0 <= asigma <= a*sigma_max
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_s,idx_a]] = 1., -sigma_max
        A_lt.append(row)
        b_lt.append(0.)
        # 0 <= sigma - asigma <= sigma_max - a*sigma_max
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_s,idx_as, idx_a]] = 1., -1., sigma_max
        A_lt.append(row)
        b_lt.append(sigma_max)
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_s,idx_as]] = 1., -1.
        A_gt.append(row)
        b_gt.append(0.)
        # a+ + a- >= asigma
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_ap,idx_am, idx_as]] = 1., -1., -1.
        A_gt.append(row)
        b_gt.append(0.)
        # z+ + z- - a+ - a- <= sigma - asigma
        row = np.zeros(ncols,dtype=np.double)
        row[[idx_p, idx_m, idx_ap, idx_am, idx_s, idx_as]] = 1., 1., -1., -1., -1., 1.
        A_lt.append(row)
        b_lt.append(0.)
        # z+ - z- + K/nu*tec + cs = phase
        row = np.zeros(ncols,dtype=np.double)
        if solve_cs:
            row[[idx_p, idx_m, idx_tec, idx_cs]] = 1., -1., K/freqs[i], 1.
        else:
            row[[idx_p, idx_m, idx_tec, idx_cs]] = 1., -1., K/freqs[i], 0.
        A_eq.append(row)
        b_eq.append(obs_phase_unwrap[i])
        # minimize z+ + z- - a+ - a- + Nsigma_max
        c_obj[[idx_p, idx_m, idx_ap, idx_am,idx_s]] = 1., 1., -1., -1.,N
    row = np.zeros(ncols,dtype=np.double)
    for i in range(N):
        idx_a = 5*N + i
        # sum a < fout * N
        row[idx_a] = 1.
    A_lt.append(row)
    b_lt.append(fout*N)
    
    A_eq, b_eq = np.array(A_eq), np.array(b_eq)
    A_lt, b_lt = np.array(A_lt), np.array(b_lt)
    A_gt, b_gt = np.array(A_gt), np.array(b_gt)
    
    
    from mippy.lpsolver import LPSolver
    lp = LPSolver(c_obj,A_eq=A_eq, b_eq=b_eq, A_lt=A_lt, b_lt=b_lt, A_gt=A_gt, b_gt=b_gt, maximize=False,problem_name=problem_name, solver_type='SIMP')
    for i in range(len(freqs)):
        idx_p = i
        idx_m = N + i
        idx_ap = 2*N + i
        idx_am = 3*N + i
        idx_as = 4*N + i
        idx_a = 5*N + i
        idx_s = 6*N
        idx_tec = 6*N + 1
        idx_cs = 6*N + 2
        lp.set_variable_type(idx_p,'c',('>',0.))
        lp.set_variable_type(idx_m,'c',('>',0.))
        lp.set_variable_type(idx_ap,'c',('>',0.))
        lp.set_variable_type(idx_am,'c',('>',0.))
        lp.set_variable_type(idx_as,'c',('>',0.))
        lp.set_variable_type(idx_a,'i',('<>',0., 1.))
        lp.set_variable_type(idx_s,'c',('<>',0., sigma_max))
        lp.set_variable_type(idx_tec,'c',('*',))
        lp.set_variable_type(idx_cs,'c',('*',))
    lp.compile()
    res = lp.submit_problem()
    for i in range(len(freqs)):
        idx_p = i
        idx_m = N + i
        idx_ap = 2*N + i
        idx_am = 3*N + i
        idx_as = 4*N + i
        idx_a = 5*N + i
        idx_s = 6*N
        idx_tec = 6*N + 1
        idx_cs = 6*N + 2
        assert np.isclose(res[idx_p]*res[idx_m], 0.) , "infeasible solution, {},{}".format(res[idx_p],res[idx_m])
    return res[[6*N, 6*N+1, 6*N+2]]

def l1_lpsolver_parallel(obs_phase, freqs, sigma_max = np.pi, fout=0.5, solve_cs=True, problem_name="l1_tec_solver",num_threads = None):
    '''Solve the tec and cs for multiple datasets.
    `obs_phase` : `numpy.ndarray`
        the measured phase with shape (num_freqs, num_datasets)
    `freqs` : `numpy.ndarray`
        the frequencies at the datapoints (num_freqs,)
    `sigma_max` : (optional) `float`
        the maximum allowed deviation for outlier detection. default np.pi
    `fout` : (optional) `float`
        The maximum fraction of allowed outliers out of total number of datapoints. default 0.5
    `solve_cs` : (optional) bool
        Whether to solve cs (True)
    `num_threads` : (optional) `int`
        number of parallel threads to run. default None is num_cpu
    `problem_name` : (optional) `str`
        name of problem "l1_tec_solver"
    '''
    from dask import delayed, compute
    from dask.threaded import get
    from functools import partial
    dsk = {}
    assert len(obs_phase.shape) == 2, "obs_phase not dim 2 {}".format(obs_phase.shape)
    N = obs_phase.shape[1]
    values = [delayed(partial(l1_lpsolver, sigma_max=sigma_max, fout=fout,solve_cs=solve_cs, problem_name="{}{:03d}".format(problem_name,i)), pure=True)( obs_phase[:,i], freqs) for i in range(N)]
    #client = Client()
    results = compute(*values, get=get, num_workers=num_threads)
    return results

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
    obs_phase = phase_unwrapp1d(obs_phase,axis=0)
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
        #return np.nansum(np.abs(K*np.multiply.outer(1./freqs,tec) - phase)/sigma_phase,axis=0)
    
        return beta*((tec - tec_p)**2/sigma_tec2 + (cs - cs_p)**2/sigma_cs2)/2 + np.nansum(np.abs(K*np.multiply.outer(1./freqs,tec) + alpha*cs - phase)/sigma_phase,axis=0)
    
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
        grad[:,0] = x0*(beta*(tec - tec_p)/x0 + np.nansum((x1*x2),axis=0))
        grad[:,1] = x3 * (beta*(cs - cs_p)/x3 + np.nansum(alpha*x2,axis=0))
        return grad
    
    def calc_epsilon_n(dir,m,freqs,CdCt,obs_phase,step=1e-3):
        """Approximate stepsize"""
        g0 = calc_phase(m, freqs)
        gp = calc_phase(m + step*dir, freqs)
        Gm = (gp - g0)/step
        dd = obs_phase - g0
        epsilon_n = (np.nansum(Gm*dd/CdCt,axis=0)/np.nansum(Gm/CdCt*Gm,axis=0))
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
    #print( np.nansum(np.abs(calc_phase(m,freqs) - obs_phase)/np.sqrt(CdCt),axis=0))
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
        #print("iter {}: neglogL: {}, log|dm/m|: {}, |grad|: {}".format(iter, S, np.mean(np.log(np.abs(np.einsum("i,ij->ij",epsilon_n,dir)/m))),np.nansum(np.abs(grad))))
        iter += 1
    #print("Final neglogL: {}".format(S))
    return m
