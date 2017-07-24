from rathings.phase_unwrap import phase_unwrapp1d
from rathings.tec_solver import l1data_l2model_solve, l1_lpsolver
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
    #one time case
    times = np.arange(1)
    freqs = np.linspace(110e6,170e6,100)
    cs = times*0
    tec = times*0 + 0.001
    phase_ = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec)
    phase = phase_ + 2*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    phase[50:70,:] +=  6*np.pi/180.*np.random.normal(size=[20,len(times)])
    
    #wrap the phsae
    phase = np.angle(np.exp(1j*phase))

    m = l1data_l2model_solve(phase,freqs,5)
    rec_phase = np.multiply.outer(np.ones(len(freqs)),m[:,1]) + 8.44797256e-7*1e16*np.multiply.outer(1./freqs,m[:,0])
    #assert np.all(np.abs(rec_phase - phase_)<0.01)
    sigma_out, tec,cs = l1_lpsolver(phase[:,0],freqs,fout=0.5)
    print("outlier thresh: {} deg, tec: {}, cs: {}".format(sigma_out*180./np.pi, tec, cs))

    rec_phase = 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec) + cs
#    import pylab as plt
#    plt.plot(freqs,phase)
#    plt.plot(freqs,rec_phase)
#    plt.show()
    print("MAE: {}".format(np.sum(np.abs(rec_phase - phase))/len(freqs)))

