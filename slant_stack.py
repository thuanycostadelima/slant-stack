#!/usr/bin

import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.core import Trace, Stream, UTCDateTime
from matplotlib import rcParams
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
from obspy.taup import TauPyModel
import sys
from obspy.io.sac.util import get_sac_reftime
import os, sys
from obspy.core import Trace
from obspy.io.sac import SACTrace
import os
import time

start_time = time.time()

def read_traces(dname):
    st = Stream()
    fname_list = glob.glob('%s/*sac'%dname)
    
    for fname in fname_list:
        tr = read(fname, format='SAC')[0]
        T0 = get_sac_reftime(tr.stats.sac) + tr.stats.sac.t6
        dt = tr.stats.sac.delta
        npts = tr.stats.sac.npts
        #print("Before trim", len(tr.data))

        tr.trim( T0 + lim_tmin , T0 + lim_tmax)
        tr.normalize()
        #print("After trim",len(tr.data))
        st.append(tr)
    
    return st

def slant_stack(data_stream, delta_rayp):

    dist = np.array([tr.stats.sac.gcarc for tr in data_stream])
    ref_dist = np.median(dist) # You might want to change this
    delta_dist = dist - ref_dist

    npts = data_stream[0].stats.npts
    nsta = len(data_stream)
    delta = data_stream[0].stats.delta

    shift_time = np.outer(delta_rayp, delta_dist)
    imag = np.zeros((len(delta_rayp), npts))

    for i_rayp in range(len(delta_rayp)):
        
        data_tr = []
        shifted_st = data_stream.copy()

        for i_trace, tr in enumerate(shifted_st):
            shift = -int(np.round(shift_time[i_rayp, i_trace] / delta))
            #data = np.zeros_like(tr.data)
            #if shift == 0:
            #    data = tr.data
            #else:
            #    data[shift:]=tr.data[:-shift]
            #data_tr.append(data )
            data_tr.append(np.roll(tr.data, shift) )
        imag[i_rayp, :] = np.sum([tr for tr in data_tr], axis=0) / nsta

    return imag
    
####################################
rcParams['font.size'] = 10
taup_model = TauPyModel('ak135')

## Get the current working directory
cwd = os.getcwd()
print(cwd)

## Defining the time window
lim_tmin = -390
lim_tmax = 50

### Extracting a cross-section from the vespagram
cross_section_file = 'SSinfo_0.1s.dat'
def extract_key(line):
  return tuple(map(float, line.split(maxsplit=2)[0:2]))

def do_proc(delta):
  with open(cross_section_file, 'r') as f:
    time_slow = frozenset(extract_key(line) for line in f)

  with open('vespa.dat', 'r') as f:
    lines = ''.join(line for line in f if extract_key(line) in time_slow)

  with open('wiggle_orig_test.dat', 'w') as f:
    print(lines, end='', file=f)

  data = np.loadtxt('wiggle_orig_test.dat', usecols=2)
  tt = np.loadtxt('wiggle_orig_test.dat', usecols=0)
  data = np.array(data)
  tt = np.array(tt)
  print(tt[0])
  header={'delta':delta, 'b': float(tt[0]),'npts': len(data), 'kstnm': 'x_section'}
  tr = SACTrace(data=data, **header)
  
  #tr.sac.b = tt[0]/0.1
  tr.write("/Users/thuanycostadelima/Desktop/%s_syn.sac" % bin_name)

if __name__ == '__main__':

    dname = sys.argv[1]
    bin_name = dname.split('/')[0]
    print("Into the sac directory: ", str(dname), str(bin_name))

    #### Defining the parameters
    delta_rayp = np.arange(-1.60, 0.01, 0.01)
    scale_factor = 0.01
    data_stream = read_traces(dname)
    imag = slant_stack(data_stream, delta_rayp)
    evdp = data_stream[0].stats.sac.evdp
    npts = data_stream[0].stats.npts
    delta = data_stream[0].stats.delta
    gcarc = np.max([tr.stats.sac.gcarc for tr in data_stream])
    delta_time = np.arange(lim_tmin, lim_tmax+0.1, 0.1)

    #arvs = taup_model.get_travel_times(evdp, gcarc, ['SS'])
    
    #### Output vespagram into a vespa.dat file
    with open('vespa.dat', 'w') as f:
        for (irow, tr1), irap in zip( enumerate(imag), delta_rayp):   
            for dtime, amp in zip(delta_time, tr1):
                f.write( str("%.1f" % dtime) + ' ' + str("%.2f" % irap)+ ' ' + str(amp) + '\n' )
    f.close()
    print("Done exporting vespa.dat...")

    #### Taking the cross-section from the vespagram
    do_proc(delta=delta)

    #### 
    x = np.loadtxt(cross_section_file, usecols=0)
    y = np.loadtxt(cross_section_file, usecols=1)
    x = np.array(x)
    y = np.array(y)

    #### Making the plot of vespagram
    #ax = plt.gca()
    fig, ax = plt.subplots(1,1,figsize=(5, 4), dpi=100, subplot_kw={'aspect': 'auto'})
    
    ax.imshow(imag, extent=(lim_tmin, lim_tmax, 0, min(delta_rayp)), interpolation='bilinear', \
        aspect='auto', vmax=np.max(imag)*scale_factor, vmin=(-1)*np.max(imag)*scale_factor, cmap='coolwarm')
    

    ax.plot( x,y, color='black', marker=None, linestyle='--', linewidth=1.5, zorder=100000)

    ax.set_xlim(-330,lim_tmax)

    ax.set_xlabel('Time relative to SS (s)')
    ax.set_ylabel('Ray parameter (s/deg)')
    ax.grid(ls='--', lw=.5, color='gray')

    ax.plot(0, 0, '+w', ms=10)
    ax.text(0, .15, 'SS', va='top', ha='center', color='black')

    plt.tight_layout()

    #### Saving the vespagram figure
    os.remove('vespa.dat')
    os.remove('wiggle_orig_test.dat')
    plt.savefig("/Users/thuanycostadelima/Desktop/%s.png" % bin_name, dip=300)
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
