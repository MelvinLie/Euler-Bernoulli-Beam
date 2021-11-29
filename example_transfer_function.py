import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import EulerBernoulliBeam as eb_beam
import os
from matplotlib.animation import FuncAnimation
from scipy import interpolate



#----------------------------------------------
# Material Parameters

#   Densities
rho_al = 2346.82   #[kg/m**3]
rho_brass = 8730 #[kg/m**3]
rho_pom = 2822.841  #[kg/m**3]  measured
rho_gfk = 1541.354

#   Elastic Modulus
E_al    = 69e9     #[Pa]
E_brass = 100e9     #[Pa]
E_pom = 2.5e9
E_gfk   = 60e9     #[Pa]


#   Measured weights
m_gfk = 62.95e-3    #[kg]
#----------------------------------------------


Beam = eb_beam.EulerBernoulliBeamAssembly()

#---------------------------------------------------
#Alu Tube
I_1 = np.pi/4*(0.02**4-0.017**4)
m_1 = 0.6547/0.8   #Measured #np.pi*(0.02**2-0.017**2)*rho_al
E_1 = E_al
segs_1 = 73
color_1 = [132./255,135./255,137./255]
L_1 = 0.8 - 0.072

pos_0 = np.array([0.,0.,0.])
pos_1 = np.array([0.,0.,L_1])

Beam.connect_knots(pos_0,pos_1,segs_1,E_1,I_1,m_1,color=color_1,linewidth=12)



#---------------------------------------------------
#Alu Tube with Delrin Reinforcement plus gfk
EI_2 = E_al*np.pi/4 * (0.02**4-0.017**4)    \
        + E_pom*np.pi/4 * (0.017**4-0.007**4) \
        + E_gfk * np.pi/4*(0.007**4-0.006**4)

m_2 = np.pi*(0.02**2-0.017**2)*rho_al   \
        + np.pi*(0.017**2-0.007**2)*rho_pom \
        + np.pi*(0.007**2-0.006**2)*rho_gfk

segs_2 = 7
color_2 = [200./255,200./255,190./255]
L_2 = 0.072

pos_2 = np.array([0.,0.,L_1+L_2])

Beam.connect_knots(pos_1,pos_2,segs_2,EI_2,1.,m_2,color=color_2,linewidth=12)

#---------------------------------------------------
#Delrin part plus gfk
segs_3 = 3
color_3 = [255./255,250./255,205./255]
L_3 = 0.007

EI_3 = E_pom * np.pi/4*(0.02**4-0.007**4)   \
        + E_gfk * np.pi/4*(0.007**4-0.006**4)

m_3 = np.pi*(0.02**2-0.007**2)*rho_pom + np.pi*(0.007**2-0.006**2)*rho_gfk

pos_3 = np.array([0.,0.,L_1+L_2+L_3])

Beam.connect_knots(pos_2,pos_3,segs_3,EI_3,1.,m_3,color=color_3,linewidth=12)

#---------------------------------------------------
#GFK tube with Delrin Reinforcement
EI_4 = E_pom * np.pi/4*(0.0115**4-0.007**4) \
        + E_gfk * np.pi/4*(0.007**4-0.006**4)

m_4 = np.pi*(0.0115**2-0.007**2)*rho_pom  + np.pi*(0.007**2-0.006**2)*rho_gfk
segs_4 = 4
color_4 = [255./255,250./255,205./255]
L_4 = 0.037

pos_4 = np.array([0.,0.,L_1+L_2+L_3+L_4])

Beam.connect_knots(pos_3,pos_4,segs_4,EI_4,1.,m_4,color=color_4,linewidth=6)

#---------------------------------------------------
#GFK tube
I_5 = np.pi/4*(0.007**4-0.006**4)
m_5 =  np.pi*(0.007**2-0.006**2)*rho_gfk
E_5 = E_gfk
segs_5 = 96
color_5 = [0.1,0.1,0.1]
L_5 = 0.835


pos_5 = np.array([0.,0.,L_1+L_2+L_3+L_4+L_5])

Beam.connect_knots(pos_4,pos_5,segs_5,E_5,I_5,m_5,color=color_5,linewidth=5)


#---------------------------------------------------
#GFK inside
I_6 = np.pi/4*(0.007**4-0.006**4)
m_6 =  np.pi*(0.007**2-0.006**2)*rho_gfk
E_6 = E_gfk
segs_6 = 5
color_6 = [0.1,0.1,0.1]
L_6 = 0.29 - L_4 - L_3  - L_2#1. - L_5 - L_4 - L_3  - L_2

pos_6 = pos_1 - np.array([0.,0.,L_6])

#this part is the overhang of the GFK tube inside the alu tube
Beam.connect_knots(pos_6,pos_1,segs_6,E_6,I_6,m_6,color=color_6,linewidth=5)

#Total mass of sensor, cable and connectors
#M_sensor = 0.012
#Beam.add_mass_between_positions(pos_0[2],pos_6[2],M_sensor/(pos_6[2] - pos_0[2]))

K = Beam.assemble_K().toarray()
p = Beam.assemble_p()

M = Beam.assemble_M().toarray()


#support boundary condition
u = [2e5,1e3,2e5,1e3]

K[6,6] += u[0]
K[7,7] += u[1]

K[68,68] += u[2]
K[69,69] += u[3]
K[70,70] += u[2]
K[71,71] += u[3]


M = Beam.assemble_M().toarray()


'''*****************************************************************************
Eigenmodes
*****************************************************************************'''
w, v = eigh(K,b=M)

#first N Eigenfrequencies
N = 5
f_eig = np.sqrt(w[:N])/2/np.pi

print("Fist {} eigenfrequencies = {} Hz".format(N,f_eig))


fig = plt.figure()
ax = fig.add_subplot(111)
Beam.plot_deformation(v[:,0],ax)
ax.grid(which='both')
ax.set_xlabel(r"$z\rightarrow$ [m]")
ax.set_ylabel(r"$w\rightarrow$")
plt.show()



'''*****************************************************************************
Transfer function
*****************************************************************************'''
z_end = np.array([1.675])
z_target = [1.500]

#Raileigh Damping
ll = 0.001
uu = 0

p = Beam.assemble_p()

num_freq = 100
max_freq = 150
min_freq = 0.25

fs = np.linspace(min_freq,max_freq,num_freq)

wend_real = 1j*np.zeros((num_freq,))
wend_imag = 1j*np.zeros((num_freq,))

tend_real = 1j*np.zeros((num_freq,))
tend_imag = 1j*np.zeros((num_freq,))


wtarget_real = 1j*np.zeros((num_freq,))
wtarget_imag = 1j*np.zeros((num_freq,))

ttarget_real = 1j*np.zeros((num_freq,))
ttarget_imag = 1j*np.zeros((num_freq,))


support_dofs = [6,7,68,69]
support_vals = [1e6,0,1e6,0]

K = Beam.assemble_K().toarray()

K[6,6] += u[0]
K[7,7] += u[1]

K[68,68] += u[2]
K[69,69] += u[3]


for i,f in enumerate(fs):

    omega = 2*np.pi*f

    pp = 0*1j*p.copy()

    #boundary conditions
    for j,dof in enumerate(support_dofs):
        pp[dof] += support_vals[j]

    A =  -omega**2*M  + 1j*omega*(ll*K + uu*M) + K

    x = np.linalg.solve(A,pp)


    wend_real[i] = Beam.compute_displacement(z_end,np.real(x) )
    wend_imag[i] = Beam.compute_displacement(z_end,np.imag(x) )


    tend_real[i] = Beam.evaluate_slope(np.real(x), z_end )
    tend_imag[i] = Beam.evaluate_slope(np.imag(x), z_end )


    wtarget_real[i] = Beam.compute_displacement(z_target,np.real(x) )
    wtarget_imag[i] = Beam.compute_displacement(z_target,np.imag(x) )

    ttarget_real[i] = Beam.evaluate_slope(np.real(x), z_target )
    ttarget_imag[i] = Beam.evaluate_slope(np.imag(x), z_target )

    #if(i == 1):
    #    print(x)

w_real_ft = interpolate.interp1d(fs, wend_real,fill_value="extrapolate")
w_imag_ft = interpolate.interp1d(fs, wend_imag,fill_value="extrapolate")

t_real_ft = interpolate.interp1d(fs, tend_real,fill_value="extrapolate")
t_imag_ft = interpolate.interp1d(fs, tend_imag,fill_value="extrapolate")

w_real_target = interpolate.interp1d(fs, wtarget_real,fill_value="extrapolate")
w_imag_target = interpolate.interp1d(fs, wtarget_imag,fill_value="extrapolate")

t_real_target = interpolate.interp1d(fs, ttarget_real,fill_value="extrapolate")
t_imag_target = interpolate.interp1d(fs, ttarget_imag,fill_value="extrapolate")

def w_transfer_function(f,Y):
    if(abs(f) > max_freq) or (abs(f) < min_freq):
        return 0.
    if (f > 0):
        return (w_real_ft(f) + 1j*w_imag_ft(f))*Y
    else:
        return (w_real_ft(-f) - 1j*w_imag_ft(-f))*Y

def t_transfer_function(f,Y):
    if(abs(f) > max_freq) or (abs(f) < min_freq):
        return 0.
    if (f > 0):
        return (t_real_ft(f) + 1j*t_imag_ft(f))*Y
    else:
        return (t_real_ft(-f) - 1j*t_imag_ft(-f))*Y


def w_transfer_function_target(f,Y):
    if(abs(f) > max_freq) or (abs(f) < min_freq):
        return 0.
    if (f > 0):
        return (w_real_target(f) + 1j*w_imag_target(f))*Y
    else:
        return (w_real_target(-f) - 1j*w_imag_target(-f))*Y

def t_transfer_function_target(f,Y):
    if(abs(f) > max_freq) or (abs(f) < min_freq):
        return 0.
    if (f > 0):
        return (t_real_target(f) + 1j*t_imag_target(f))*Y
    else:
        return (t_real_target(-f) - 1j*t_imag_target(-f))*Y


'''*****************************************************************************
plot transfer function
*****************************************************************************'''
f = np.linspace(0,150,1000)
Y = np.ones((1000,)) + 0j

tf = np.array([w_transfer_function(ff,Y[i]) for i,ff in enumerate(f)])
tf_t = np.array([t_transfer_function(ff,Y[i]) for i,ff in enumerate(f)])



fig = plt.figure()
ax1 = fig.add_subplot(211)
#ax.plot(w_real_ft(f))
#ax.plot(w_imag_ft(f))
ax1.plot(f,abs(np.real(tf)),label='real')
ax1.plot(f,abs(np.imag(tf)),'--',label='imag')
ax1.set_xlabel(r"$f\rightarrow$ [1/s]")
ax1.set_ylabel(r"$T_w(f)\rightarrow$")
ax1.grid(which='both')
ax1.legend()
ax1.set_yscale('log')
#tikzplotlib.save("transfer_function_w.tex",axis_height="6cm",axis_width="14cm")

ax2 = fig.add_subplot(212)
#ax.plot(w_real_ft(f))
#ax.plot(w_imag_ft(f))
ax2.plot(f,abs(np.real(tf_t)),label='real')
ax2.plot(f,abs(np.imag(tf_t)),'--',label='imag')
ax2.set_xlabel(r"$f\rightarrow$ [1/s]")
ax2.set_ylabel(r"$T_\theta(f)\rightarrow$")
ax2.grid(which='both')
ax2.legend()
ax2.set_yscale('log')
ax2.set_ylim(ax1.get_ylim())
#tikzplotlib.save("transfer_function_t.tex",axis_height="6cm",axis_width="14cm")
plt.show()
