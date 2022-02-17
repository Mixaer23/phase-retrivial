import numpy as np
import xraylib
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Includes
def field( wl, tht, STR ):
    """ Calculate reflectance and propagation matrix based on RT formalism """
    propagator = []
    kz = 2*np.pi/wl * np.sqrt(STR[:,1]**2 - STR[0,1]**2 * np.cos(tht)**2);
    p = (kz[:-1] + kz[1:]) /2/kz[:-1]                   
    m = (kz[:-1] - kz[1:]) /2/kz[:-1]                   

    M = np.eye(2)
    for j in range(STR.shape[0]-1):   
        phi  = np.exp( -1j*kz[j]*STR[j,0]  )
        phi_ = np.exp( +1j*kz[j]*STR[j,0]  )
        T = np.array(  [[ phi_, 0 ],
                        [   0, phi]]  )
        R = np.array(  [[ p[j], m[j] ],
                        [ m[j], p[j]]]  )
        TR = T @ R
        M = M @ TR
        propagator.append(TR)

    r = M[0,1]/M[1,1]
    RT = [(r,1)]
    for TR in propagator:
        RT.append( np.linalg.inv(TR) @ RT[-1] )
    return np.array(RT), kz

def energy(wl : float) -> float:
    """converts nm to keV"""
    return 1.23984193/wl 

def fermi(x, loc=0, sig=1):
    """ Fermi fucntion """
    return 1/( np.exp((-x+loc)/sig) + 1  )

def jump(x,loc,sig,a,b):
    """ Jump at the interface """
    f = fermi(x,loc,sig)*(b-a)+a
    return f

def from_str( x, S ):
    """ Generate profile """
    S = np.array(S)
    Z = np.cumsum(S[:,0])
    f = 0
    for j in range(len(S)-1):
        a = S[j][1]-1
        b = S[j+1][1]-1
        z = Z[j]
        sig = S[j+1][2]
        f += jump(x,z,sig,0,b-a) 
    return f+1

def xrr( z, tht, wl, struc ):
    """ Calculate field for smooth profile"""
    profile = from_str(z, struc)
    struc = np.block([[profile],[profile]]).T
    struc[:,0] = z[1]-z[0]

    r = []
    NF = []
    for tht_i in tht:
        E,_ = field( wl, tht_i, struc )
        r.append(E[0,0])
        NF.append(E)
    r = np.array(r)
    NF = np.array(NF)
    return r, NF, profile

def xrr_exp( z, tht, wl, load ):
    """ Calculate field for smooth profile"""
    profile = load[:,1]
    profile = np.interp(z,load[:,0].real,profile)
    struc = np.block([[profile],[profile]]).T
    struc[:,0] = (z[1]-z[0])

    r = []
    NF = []
    for tht_i in tht:
        E,_ = field( wl, tht_i, struc )
        r.append(E[0,0])
        NF.append(E)
    r = np.array(r)
    NF = np.array(NF)
    return r, NF, profile

def phase_rm(r,qz,q_z_c):
    """ Removes Phase"""
    fi = np.angle(r)
    k = 0
    for i in range(len(qz)):
        if qz[i] < q_z_c:
            k = i
    fi = np.resize(fi,k)
    for i in range(r.size-k):
        fi = np.append(fi,0)
    r = np.abs(r) * np.exp(1j*fi)
    return r

def HIO(zs_min,zs_max,beta,gk,gk_c): #values, min max support, function
    """ HIO Phase retrivial """
    gk1 = gk_c
    gk_c = abs( gk ) - abs( beta*gk_c )
    gk_c[zs_min:zs_max:] = gk1[zs_min:zs_max:]
    return gk_c.real

def CHIO(zs_min,zs_max,alfa,beta,gk,gk_c): 
    """ CHIO Phase retrivial """
    for k in range(gk.size):
        if  (k >= zs_min) and(k < zs_max) :
            gk_c[k] = gk_c[k]
        elif gk_c[k].real < alfa * gk[k].real:
            gk_c[k] = abs( gk[k] ) - abs( ((1-alfa)/alfa) * gk_c[k] )
        else:
            gk_c[k] = abs( gk[k] ) - abs( beta * gk_c[k] )
    lim = gk_c.max()
    for ii in range( len( gk ) ):   #Binarization
        if abs( gk_c[ ii ] ) < 0.15 * abs( lim ) and  ( ii % 40 >=0 ) and ( ii % 40 <= 10 ) :
            gk_c[ ii ] = 0
#        if abs( gk_c[ ii ] ) < 0.15 * abs( lim ) :
#            gk_c[ ii ] = 0
    return gk_c.real

def z_pad(arr,pad):
    """ Zero Padding """
    if arr.size > pad:
        return arr
    else:
        arr = np.append(arr,np.zeros(pad-arr.size))
        return arr
         
def ER(zs_min,zs_max,beta,gk,gk_c): #values, min max support, function
    "ER Phase retrivial"
    gk1 = gk_c
    gk_c = np.zeros(gk.size)
    gk_c[zs_min:zs_max:] = gk1[zs_min:zs_max:].real
    return gk_c

def crit_approx(profile,wl,q_z):
    """ Approximation to critical angle """
    q_z_c = q_z
    tht_c = (2*(1-np.sum(profile.real)/profile.size))**0.5
    q_z_crit = 4 * np.pi / wl * np.sin(tht_c)
    q_z_c = np.sqrt(q_z**2 - q_z_crit**2, dtype = complex)
    return q_z_c,tht_c,q_z_crit

def phase_retrivial( F_d, drho_d, percent ):
    N_cycle = 1
    max_iter = int( 1e+5 )
    aver = np.zeros( ( len( drho_d ), max_iter ) ) 
    beta = 0.9
    alfa = 0.4
    k_c_percent = percent
    sup_min = 0
    sup_max = 125
    k_cut = 0
    F = np.conj( F_d )
    k_c = int( k_c_percent * F.size )
    for tt in (range( N_cycle ) )  :
        phs0 = np.pi * ( 2 * np.random.random( F.size ) - 1 )
        phs_t = np.angle( F )
        Gk0 = np.zeros( len( F ), dtype = complex )
        Gk0[k_cut::] = np.abs( F[k_cut::] ) * np.exp( 1j * phs0[k_cut::] )
        Gk0[ : k_c : ] = np.abs( Gk0[ : k_c : ]) * np.exp( 1j * phs_t[ : k_c : ] )
        gk0 = np.fft.irfft( Gk0 )
        gk = gk0
        for ii in tqdm( range( np.int( max_iter ) ) ) :   
            Gk = np.fft.rfft( gk )
            Gk_c = Gk
            Gk_c[k_cut::] = np.abs( F[k_cut::] ) * np.exp( 1j * np.angle( Gk[k_cut::] ) )
#            if ii % 40 == 0:
#                Gk_c[ : k_c : ] = abs(Gk[ : k_c : ]) * np.exp( 1j * phs_t[ : k_c : ] )       
            gk_c = np.fft.irfft( Gk_c )
            gk_next = CHIO( sup_min, sup_max, alfa, beta, gk, gk_c )
            gk = gk_next            
            aver[ :, ii ] = gk_next
    return aver, Gk

