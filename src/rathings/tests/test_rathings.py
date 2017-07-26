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
    #one time case
    times = np.arange(2)
    freqs = np.linspace(110e6,170e6,1000)
    cs = times*0
    tec = times*0 + 0.001
    phase_ = np.multiply.outer(np.ones(len(freqs)),cs) + 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec)
    phase = phase_ + 4*np.pi/180.*np.random.normal(size=[len(freqs),len(times)])
    phase[50:70,:] +=  8*np.pi/180.*np.random.normal(size=[20,len(times)])
    
    #wrap the phsae
    phase = np.angle(np.exp(1j*phase))

    m = l1data_l2model_solve(phase,freqs,5)
    rec_phase = np.multiply.outer(np.ones(len(freqs)),m[:,1]) + 8.44797256e-7*1e16*np.multiply.outer(1./freqs,m[:,0])
    #assert np.all(np.abs(rec_phase - phase_)<0.01)
    sigma_out, tec1,cs1 = l1_lpsolver(phase[:,0],freqs,fout=0.5)
    print("outlier thresh: {} deg, tec: {}, cs: {}".format(sigma_out*180./np.pi, tec1, cs1))
    sigma_out, tec2,cs2 = l1_lpsolver(phase[:,1],freqs,fout=0.5)
    #print("outlier thresh: {} deg, tec: {}, cs: {}".format(sigma_out*180./np.pi, tec2, cs2))

    sigma_out, tec3,cs3 = l1_lpsolver(phase[:,0],freqs,fout=0.5, solve_cs=False)
    print("(no cs) outlier thresh: {} deg, tec: {}, cs: {}".format(sigma_out*180./np.pi, tec3, cs3))

    rec_phase = 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec1) + cs1


    rec_phase_nocs = 8.44797256e-7*1e16*np.multiply.outer(1./freqs,tec3) + cs3
    import pylab as plt
    plt.plot(freqs,phase[:,0],label='data')
    plt.plot(freqs,rec_phase, label='with cs')
    plt.plot(freqs,rec_phase_nocs, label='without cs')
    plt.legend(frameon=False)
    plt.show()
    print("MAE: {}".format(np.sum(np.abs(rec_phase - phase[:,0]))/len(freqs)))
    tec_val = [tec1,tec2]
    for (sigma_out, tec,cs),t_val in zip(l1_lpsolver_parallel(phase,freqs,fout=0.5),tec_val):
        print("outlier thresh: {} deg, tec: {}, cs: {}".format(sigma_out*180./np.pi, tec, cs))
        assert tec==t_val
    
    tec1,cs1 = robust_l2(phase[:,0],freqs)
    print("(robust l2) tec: {}, cs: {}".format(tec1,cs1))
    tec2,cs2 = robust_l2(phase[:,1],freqs)
    print("(robust l2) tec: {}, cs: {}".format(tec2,cs2))

    tec_val = [tec1,tec2]
    for (tec,cs),t_val in zip(robust_l2_parallel(phase,freqs),tec_val):
        print("tec: {}, cs: {}".format(tec, cs))
        assert tec==t_val


    
