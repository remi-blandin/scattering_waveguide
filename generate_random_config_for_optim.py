import torch
import numpy as np
from scipy.constants import c  # Speed of light in vacuum
import scattering_waveguide_functions as fct
import csv

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
    'use_precomputed_values': True,                         # if set to true, precomputed data are used to accelerate computations
    }
    
# compute the number of propagating modes
nu_c = c / (2 * parameters['W_guide'])  # Cut-off frequencies
kn = (2 * np.pi / c) * torch.rsqrt(parameters['frequency']**2 - (torch.arange(1, 1001) * nu_c)**2)  # Waveguide mode k parameters
parameters['Nport'] = torch.max(torch.nonzero(torch.imag(kn) == 0)) + 1  # Number of modes of the empty waveguide (S is a matrix of dimension 2Nx2N)
print(f"Number of propagating modes: {parameters['Nport']}")

#%% Scatterers parameters


alphas_metal = -1j * 6      # polarizability
nb_scat = 3                # number of scatterers
radius = 0.0031              # radius of the scatterers

#%% Generate a random configuration

# first the polarization vector is initialized with the teflon property
parameters['alphas0'] = torch.full((nb_scat,), alphas_metal, dtype=torch.complex64)

# the location of the scatterers is randomly initialized
parameters['scatRad'] = torch.full((nb_scat,), radius)

# put scatterers at random positions
parameters, scat_pos = fct.position_scatterer_random(parameters, \
                        show_progress=True)
    
#%% Plot scattering matrix

freq = torch.tensor(7.0e9,dtype=torch.cfloat)
loss,S,R = fct.calculate_loss(parameters, freq)
fct.plot_optimization_progress(parameters, loss, S.detach())

#%% Save scatterers position

# Specify the filename
filename = 'scatterers_coord.csv'

# Writing to the CSV file
with open(filename, mode='w') as file:
    writer = csv.writer(file, delimiter=';')
    # Write the header
    writer.writerow(['x', 'y'])
    # Write the coordinates
    writer.writerows(scat_pos.numpy())

print(f"Coordinates saved to {filename}")