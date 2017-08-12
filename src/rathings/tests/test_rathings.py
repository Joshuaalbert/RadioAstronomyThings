from rathings.phase_unwrap import phase_unwrapp1d
from rathings.tec_solver import l1data_l2model_solve, l1_lpsolver, l1_lpsolver_parallel, robust_l2, robust_l2_parallel
import numpy as np

def test_phase_unwrapp1d():
    peak = 30
    N = 100
    assert peak/(N-1) < np.pi, "choose a valid test case"
    phase_true = np.linspace(0,peak,N) + np.random.uniform(size=N)*(np.pi - peak/(N-1))
    phase_wrap = np.angle(np.exp(1j*phase_true))
    phase_unwrap = phase_unwrapp1d(phase_wrap)
    assert np.allclose(phase_true,phase_unwrap)
    phase_true = np.multiply.outer(phase_true,[1,0.9,0.5])
    phase_wrap = np.angle(np.exp(1j*phase_true))
    phase_unwrap = phase_unwrapp1d(phase_wrap,axis=0)
    assert np.allclose(phase_true,phase_unwrap)

def test_tec_solver():
    np.random.seed(1234)
    solvers = {#"l1data_l2model":lambda phase,freqs,solve_cs: l1data_l2model_solve(phase,freqs,15), 
"robust_l2":lambda phase,freqs,solve_cs: robust_l2(phase,freqs,solve_cs=solve_cs), "l1_lp":lambda phase,freqs,solve_cs: l1_lpsolver(phase,freqs,fout=0.5,solve_cs=solve_cs)[1:]}
                
    #one time case
    times = np.arange(2)
    freqs = np.linspace(110e6,170e6,256)
    cs = np.zeros(2)
    tec = np.array([0.1,0.1])
    phase_ = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec)
    phase = phase_ + 4*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    phase[len(freqs)>>2:len(freqs)>>1,:] +=  8*np.pi/180.*np.random.normal(size=[len(freqs)>>2,len(times)])
    
    #wrap the phase
    phase = np.angle(np.exp(1j*phase))
    import pylab as plt
    from time import clock
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(1,1,1)
    #ax2 = f.add_subplot(2,1,2)
    ax.plot(freqs,phase[:,0],ls='--',label="data")
    for meth in solvers.keys():
        t1 = clock()
        tec,cs = solvers[meth](phase[:,0],freqs,False)
        dt = clock() - t1
        rec_phase = 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec) + cs
        ax.plot(freqs,rec_phase,ls='--',label="{} (no cs)".format(meth))
        print("{} (no cs) tec {:.4f} cs {:.2f} time {}".format(meth,tec,np.angle(np.exp(1j*cs)),dt))
    for meth in solvers.keys():
        t1 = clock()
        tec,cs = solvers[meth](phase[:,0],freqs,True)
        dt = clock() - t1
        rec_phase = 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec) + cs
        ax.plot(freqs,rec_phase,label="{}".format(meth))
        print("{} tec {:.4f} cs {:.2f} time {}".format(meth,tec,np.angle(np.exp(1j*cs)),dt))

    plt.legend(frameon=False)
    plt.show()
    m1 = l1_lpsolver_parallel(phase,freqs,fout=0.5,solve_cs=False)
    m2 = robust_l2_parallel(phase,freqs,solve_cs=False)
    print(m1,m2)
