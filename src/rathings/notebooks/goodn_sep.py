
# coding: utf-8

# In[ ]:


from rathings import tec_solver
import h5py
import pylab as plt
import numpy as np

data = "../../../goods-n.hdf5"

h = h5py.File(data,'r')
freqs = h['freq'][...]
ants = h['ant'][...]
times = h['times'][...]

def calc_phase(tec, freqs, cs = 0.):
    '''Return the phase  from tec and CS.
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

def print_attrs(name, obj):
    #if "facet_patch" not in name or 'dir' in name:
    #    return
    if not isinstance(obj,h5py.Group):
        return
    print("Solving {}".format(name))
    phase = obj['phase'][...]
    phase[phase==0] = np.nan
    phase = phase - phase[4,...]
    ni = phase.shape[0]
    f = plt.figure(figsize=(7*4,7*4))
    
    res = tec_solver.l1_lpsolver_parallel(phase[-1,::5,:],freqs[::5],fout=0.5,solve_cs=True)
    #res = tec_solver.robust_l2_parallel(phase[:,::5,0].T,freqs[::5],solve_cs=True)

    for i,(sigma_out, TEC, CS) in zip(range(ni),res):
        print("Solution to {}: sigma outlier thresh {}, tec {} cs {}".format(ants[i],sigma_out*180./np.pi,TEC,CS))
        #m = tec_solver.l1data_l2model_solve(phase[i,:,:1],freqs,5,CS_solve=False,m0=np.array([[0.,0.]]))
        ax = plt.subplot(7,7,i+1)
        ax.plot(freqs,phase[-1,:,i])
        ax.set_ylim(-np.pi,np.pi)
        rec_phase = np.angle(np.exp(1j*calc_phase(TEC,freqs,cs=CS)))# calc_phase(m[0,0],freqs,cs=m[0,1]
        ax.plot(freqs,rec_phase)
        ax.set_title(str(ants[-1]))
    plt.savefig("{}_timestamp{}_robust_l2.pdf".format(name,1),format='pdf')
    plt.show()
    
    
def plot_along_time(name, obj, start_time=0, stop_time=49, reference_ant = b'CS005HBA0'):
    #if "facet_patch" not in name or 'dir' in name:
    #    return
    assert stop_time > start_time
    if not isinstance(obj,h5py.Group):
        return
    ref_ant_idx = 0
    for i in range(len(ants)):
        if ants[i] == reference_ant:
            ref_ant_idx = i
            break
    print("Solving {}".format(name))
    phase = obj['phase'][...]
    phase[phase==0] = np.nan
    phase = phase - phase[ref_ant_idx,...]
    nant = phase.shape[0]
    nfreq = phase.shape[1]
    ntime = phase.shape[2]
    N = stop_time - start_time
    n_per_axis = np.ceil(np.sqrt(nant))
    f1 = plt.figure(figsize=(n_per_axis*4,n_per_axis*3))
    f2 = plt.figure(figsize=(n_per_axis*4,n_per_axis*3))
    for i in range(nant):
        res = tec_solver.l1_lpsolver_parallel(phase[i,::5,start_time:stop_time],freqs[::5],fout=0.5,solve_cs=True,num_threads=16)
        ax = f1.add_subplot(n_per_axis,n_per_axis,i+1)
        tecs = [tec for (_,_,tec) in res]
        ax.plot(times[start_time:stop_time], tecs)
        ax.set_title(str(ants[i]))
        for j,(sigma_out, TEC, CS) in zip(range(ntime),res):
            print("Solution to {} {}: sigma outlier thresh {}, tec {} cs {}".format(ants[i],j,sigma_out*180./np.pi,TEC,CS))
#             #m = tec_solver.l1data_l2model_solve(phase[i,:,:1],freqs,5,CS_solve=False,m0=np.array([[0.,0.]]))
            
#             ax.plot(freqs,phase[-1,:,i])
#             ax.set_ylim(-np.pi,np.pi)
#             rec_phase = np.angle(np.exp(1j*calc_phase(TEC,freqs,cs=CS)))# calc_phase(m[0,0],freqs,cs=m[0,1]
#             ax.plot(freqs,rec_phase)
            
#     plt.savefig("{}_timestamp{}_robust_l2.pdf".format(name,1),format='pdf')
    plt.show()

from functools import partial
h.visititems(partial(plot_along_time,start_time=0,stop_time=10))


# ### 

# In[ ]:




