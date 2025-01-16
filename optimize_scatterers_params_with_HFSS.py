import torch
import numpy as np
from scipy.constants import c  # Speed of light in vacuum
import scattering_waveguide_functions as fct
import csv
import skrf as rf
import matplotlib.pyplot as plt

parameters = {
    'W_guide': torch.tensor(0.1),                           # Waveguide width
    'H_guide': torch.tensor(0.4),                           # Waveguide length
    'epsr': torch.tensor(2.1),                              # Relative permittivity
    'frequency': torch.tensor(7.0e9,dtype=torch.cfloat),                         # Frequency (can be a vector)
    'posx': torch.empty(0),                                 # x coordinate of the scatterers
    'posy': torch.empty(0),                                 # y coordinate of the scatterers
    'spacing_min': torch.tensor(0.001),                     # Minimal distance to the middle
    'N_mode_use': 23,                                       # Total number of modes taken into account in the calculation of the Green function
    'n_s': torch.tensor(8),                                 # number of dipoles to model one cylinder
    'method': 'random',                                     # Optimization method (random in this case)
    'max_nb_iterations':1000,                               # max number of optimization iterations
    'learning_rate': 0.001,                                 # learning rate for Adam optimization algorithm
    'maximal_loss': 0.01,                                   # optimization stops when loss is lower 
    'use_precomputed_values': False,                         # if set to true, precomputed data are used to accelerate computations
    }
    
# compute the number of propagating modes
nu_c = c / (2 * parameters['W_guide'])  # Cut-off frequencies
kn = (2 * np.pi / c) * torch.rsqrt(parameters['frequency']**2 - (torch.arange(1, 1001) * nu_c)**2)  # Waveguide mode k parameters
parameters['Nport'] = torch.max(torch.nonzero(torch.imag(kn) == 0)) + 1  # Number of modes of the empty waveguide (S is a matrix of dimension 2Nx2N)
print(f"Number of propagating modes: {parameters['Nport']}")

#%% Scatterers parameters


alphas_metal = -1j * 6      # polarizability
radius = 0.0031              # radius of the scatterers

#%% Get the coordinates used for the HFSS simulations

filename = 'scatterers_coord_HFSS.csv'
coordinates = []
with open(filename, mode='r') as file:
    reader = csv.reader(file, delimiter=';')
    # Read each row and convert to tuple of floats
    for row in reader:
        # Convert the values to float and store as a tuple
        coordinates.append((float(row[0]), float(row[1])))

nb_scat = len(coordinates)

# update the number of scatterers
parameters['nb_scat'] = nb_scat

parameters['posx'] = torch.empty(nb_scat)
parameters['posy'] = torch.empty(nb_scat)

for idx, coord in enumerate(coordinates):
    parameters['posx'][idx] = coord[1]
    parameters['posy'][idx] = coord[0]

# first the polarization vector is initialized with the teflon property
parameters['alphas0'] = torch.full((nb_scat,), alphas_metal, dtype=torch.complex64)

# the location of the scatterers is randomly initialized
parameters['scatRad'] = torch.full((nb_scat,), radius)

#%% Plot scattering matrix 

freq = torch.tensor(7.0e9,dtype=torch.cfloat)
loss,S,R = fct.calculate_loss(parameters, freq)
fct.plot_optimization_progress(parameters, loss, S.detach())



#%% Load the scattering matrix simulated by HFSS

file_name_HFSS_scat_mat = 'scat_mat_HFSS.s8p'
network = rf.Network(file_name_HFSS_scat_mat)

freas_HFSS = network.f
index = np.argmin(np.abs(freas_HFSS - freq.numpy()))
S_HFSS = network.s[index, :, :]


alpha_var = torch.arange(-10., -1., 1.) * 1j
nb_alpha = len(alpha_var)
record = torch.zeros(nb_alpha)

for it, al in enumerate(alpha_var):
    
    # vary the polarizability
    parameters['alphas0'] = torch.full((nb_scat,), al, dtype=torch.complex64)
    
    # compute scattering matrix
    loss,S,R = fct.calculate_loss(parameters, freq)
    
    # compute difference
    S_diff = S_HFSS - S.numpy()
    record[it] = np.sum(np.abs(S_diff))

    # plot the scattering matrices at the selected frequency freq
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    im1 = ax1.imshow(np.abs(S.numpy()))
    ax1.set_title('Dipole model')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(np.abs(S_HFSS))
    ax2.set_title('HFSS')
    plt.colorbar(im2, ax=ax2)
    
    im3 = ax3.imshow(np.abs(S_diff))
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)
    
    plt.show()
    
# plot difference with respect to polarizability
plt.plot(np.imag(alpha_var), record)
plt.show()

