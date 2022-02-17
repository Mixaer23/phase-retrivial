import numpy as np
import matplotlib.pyplot as plt
from pyprx import *

#%% Structure, nm
r_e = 2.814e-2 # why there is 10^-3???????????????????   Should be 10^-2
deg = np.pi / 180
wl = 0.145 
n0 = 1
nCr = xraylib.Refractive_Index("Cr", 8, 7.19)
nSi = xraylib.Refractive_Index("Si", 8, 2.33)
nFe = xraylib.Refractive_Index("Fe", 8 , 7.87)
l0 = 3.0    #3.0
l1 = 10.0    #5.0
l2 = 10.0   #10.0
l3 = 20.0   #20.0
l4 = 5.0    #5.0

struc = [  
            [ l0,   n0,    .3],
            [ l1,   nCr,   .3],
            [ l2,   nSi,   .3],
            [ l3,   nFe,   .3],
            [ l4,   nSi,   .3],
        ]

struc = np.array( struc )
res = 500
eps = 1e-5
pad = 1500
tht = np.linspace( eps, 5, res ) * deg
z = np.linspace( 0, np.sum( struc[:,0]).real, res )

#%% XRR
r_d, fld, profile = xrr( z, tht, wl, struc )
edp = 2 * np.pi / ( wl ** 2 * r_e ) * ( 1 - profile.real )   #why there is an 10^-4??????????????????????????????/

#%% Calculation
q_z = 4 * np.pi / wl * np.sin(tht)
drho = np.gradient( edp, z[1] - z[0])
k_s = 2 * np.pi / wl * np.sqrt( struc[-1,1] ** 2 - np.cos(tht) ** 2, dtype = complex)
k_v = q_z / 2
r_f = abs( ( k_v - k_s ) / ( k_v + k_s ) )

#%% Kinematic
drho = z_pad( drho, pad )
z_k = np.linspace( z[0], z[-1] / z.size * drho.size, drho.size )
q_z_k = np.linspace( eps, np.pi / ( z_k[1] - z_k[0] ), drho.size)
F_k = np.fft.rfft( drho, n = ( drho.size - 1 ) * 2 )
F_k = np.interp( q_z, q_z_k, F_k )
q_z_c, _, q_z_crit = crit_approx( profile, wl, q_z ) 
q_z_crit = 0.6
r_k = r_f * np.interp( q_z_c.real, q_z, F_k)

#%% Dynamic
F_d = r_d / r_f
F_d = np.interp( q_z, q_z_c.real, F_d )
drho_d = np.fft.irfft( np.conj( F_d ) )
z = np.linspace( 0, 2 * np.pi / ( q_z[1] - q_z[0] ), len( drho_d  ) )

#%% Phase retrivial. Dynamic
percent = 3 * q_z_crit / q_z[-1]
aver, Gk = phase_retrivial( F_d, drho_d, percent )

#%% Saving the results
np.save("profile_reconstruction", np.mean( aver, axis = 1 ) )
np.save("profile_dynamic", drho_d )
np.save("formfactor_reconstruction", Gk )
np.save("r_dynamic", r_d )
np.save("r_kinematic", r_k )
np.save("formfactor_kinematic", F_k )
np.save("formfactor_dynamic", F_d )
np.save("q_z", q_z/10 )
np.save("z", z*10 )
np.save("z_k", z_k*10 )
