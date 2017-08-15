

from rathings.tec_solver import robust_l2_parallel, robust_l2
import h5py
import pylab as plt
import numpy as np
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

import sys

if sys.hexversion >= 0x3000000:
    def str_(s):
        return str(s,'utf-8')
else:
    def str_(s):
        return str(s)

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

def print_name(name, obj):
    '''Print all parts of hdf5. Use with visititems.'''
    #if "facet_patch" not in name or 'dir' in name:
    #    return
    #if not isinstance(obj,h5py.Group):
    #    return
    print("{}".format(name))      
    
def plot_along_time(name, obj, freqs=None,start_time=0, stop_time=49, reference_ant = 'CS005HBA0'):
   """Plot the phase for each antenna and time stamps between start_time and stop_time"""
    #if "facet_patch" not in name or 'dir' in name:
    #    return
    assert stop_time > start_time or stop_time == -1
    if not isinstance(obj,h5py.Group):
        return
    print("Solving {}".format(name))
    phase = obj['phase'][...]
    phase[phase==0] = np.nan
    nant = phase.shape[0]
    nfreq = phase.shape[1]
    ntime = phase.shape[2]
    N = stop_time - start_time
    n_per_axis = np.ceil(np.sqrt(nant))
    f1 = plt.figure(figsize=(n_per_axis*4,n_per_axis*3))
    for i in range(nant):
        #res = tec_solver.l1_lpsolver_parallel(phase[i,::5,start_time:stop_time],freqs[::5],fout=0.5,solve_cs=True,num_threads=16)
        ax = f1.add_subplot(n_per_axis,n_per_axis,i+1)
        #tecs = [tec for (_,tec,_) in res]
        ax.plot(freqs,phase[i,:,start_time:stop_time])
        #ax.plot(times[start_time:stop_time], tecs)
    plt.show()

def solve_patch(name, obj, freqs = None, data_dict=None, start_time=0, stop_time=49, reference_ant = 'CS005HBA0',num_threads = None):
    '''The function that gets run on each group object in hdf5. Solves dtec and puts into data_dict for creation of datapack. Used with visititems and partial.'''
    #if "facet_patch" not in name or 'dir' in name:
    #    return
    assert stop_time > start_time or stop_time == -1
    if not isinstance(obj,h5py.Group):
        return
    print("Solving {}".format(name))
    phase = obj['phase'][...]#ant,freq,time
    phase[phase==0] = np.nan
    nant = phase.shape[0]
    nfreq = phase.shape[1]
    ntime = phase.shape[2]
    if stop_time==-1:
        stop_time = ntime
    dir = obj['dir']#icrs pointing
    data_dict['directions'].append(dir)
    data_dict['patch_names'].append(name)
    dtec = np.zeros([nant,ntime],dtype=float)
    for i in range(nant):
        res = robust_l2_parallel(phase[i,:,start_time:stop_time],freqs[:],solve_cs=True,num_threads=num_threads)
        tecs = [tec for (tec,_) in res]
        dtec[i,:] = np.array(tecs)
    data_dict['dtec'].append(dtec)
          
def solve_dtec(data_file,datapack_file=None,start_time=0,stop_time=-1,reference_antenna='CS005HBA0',num_threads = None):
    '''Create the datapack of the entire observation stored in data_file.
    data_file is created by the transfer2hdf5 program which turns a facet
    calibration into managable data format. 
    Creates a datapack in the same directory unless datapack_file is given.
    start_time, end_time give the indices along timeaxis to do solve on.
    Use 0 and -1 for full time. Reference_antenna is the name of lofar 
    station to use as reference.'''
    from ionotomo.astro.real_data import DataPack
    from ionotomo.astro.radio_array import RadioArray
    h = h5py.File(data_file,'r')
    freqs = h['freq'][...]
    ants = h['ant'][...]
    times = at.Time(h['times'][...],format='mjd',scale='tai')
    radio_array = RadioArray(array_file=RadioArray.lofar_array)
    order = []
    for lab in ants:
        idx = radio_array.get_antenna_idx(str_(lab))
        order.append(idx)
    antennas = radio_array.get_antenna_locs()[order]
    antenna_labels = [str_(s) for s in ants]
    data_dict = {'radio_array':radio_array,'antennas':antennas,'antenna_labels':antenna_labels,'times':times,'timestamps':times.isot,'directions':[],'patch_names':[],'dtec':[]}
    from functools import partial
    #h.visititems(partial(plot_along_time,freqs=freqs,start_time=start_time,stop_time=stop_time))
    h.visititems(partial(solve_patch,freqs=freqs,data_dict=data_dict,start_time=start_time,stop_time=stop_time,num_threads =num_threads))
    dirs = np.array(data_dict['directions'])
    dirs = ac.SkyCoord(dirs[:,0]*au.rad,dirs[:,1]*au.rad,frame='icrs')
    data_dict['directions'] = dirs
    data_dict['dtec'] = np.stack(data_dict['dtec'],axis=-1)
    datapack = DataPack(data_dict=data_dict)
    try:
        datapack.set_reference_antenna(reference_antenna)
    except:
        datapack.set_reference_antenna(antenna_labels[0])
    if datapack_file is None:
        datapack.save(data_file.replace('.hdf5','-datapack.hdf5'))
    else:
        datapack.save(datapack_file)

if __name__=='__main__':
    solve_dtec('../../../goods-n.hdf5',start_time=0,stop_time=20,num_threads = 16)    
