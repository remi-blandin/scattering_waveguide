import torch
import numpy as np
from scipy.constants import c  # Speed of light in vacuum
import scattering_waveguide_functions as fct

# number and type of scatterers
nb_opt_metal = 6       # Metal scatterers to optimize
nb_opt_dielec = 5      # Teflon scatterers to optimize
nb_scat = nb_opt_metal + nb_opt_dielec  # Number of dipoles to be optimized (the nb_opt at the left part)

parameters = {
    'W_guide': torch.tensor(0.1),                           # Waveguide width
    'H_guide': torch.tensor(0.4),                           # Waveguide length
    'epsr': torch.tensor(2.1),                              # Relative permittivity
    'frequency': torch.tensor(7.0e9,dtype=torch.cfloat),                         # Frequency (can be a vector)
    'nb_scat': nb_scat,                                     # number of scatterers
    'scatRad': torch.full((nb_scat,), 0.0031),              # Radius of metallic scatterers
    'spacing_min': torch.tensor(0.001),                     # Minimal distance to the middle
    'N_mode_use': 23,                                       # Total number of modes taken into account in the calculation of the Green function
    'n_s': torch.tensor(8),                                 # number of dipoles to model one cylinder
    'method': 'random',                                     # Optimization method (random in this case)
    'max_nb_iterations':1000,                               # max number of optimization iterations
    'learning_rate': 0.001,                                 # learning rate for Adam optimization algorithm
    'maximal_loss': 0.01,                                   # optimization stops when loss is lower 
    }
    
# compute the number of propagating modes
nu_c = c / (2 * parameters['W_guide'])  # Cut-off frequencies
kn = (2 * np.pi / c) * torch.rsqrt(parameters['frequency']**2 - (torch.arange(1, 1001) * nu_c)**2)  # Waveguide mode k parameters
parameters['Nport'] = torch.max(torch.nonzero(torch.imag(kn) == 0)) + 1  # Number of modes of the empty waveguide (S is a matrix of dimension 2Nx2N)
print(f"Number of propagating modes: {parameters['Nport']}")

# polarizabilities
alphas_metal = -1j * 6
alphas_teflon = +1j * 0.048

#%% generate scatterers polarization and position

# the metal scatterers are placed at random positions in the polarization vector
index_metal = torch.randperm(nb_scat)[:nb_opt_metal]
# first the polarization vector is initialized with the teflon property
parameters['alphas0'] = torch.full((nb_scat,), alphas_teflon, dtype=torch.complex64)
# then the indexes corresponding to metal are changed to the metal property
parameters['alphas0'][index_metal] = torch.full((nb_opt_metal,), alphas_metal, dtype=torch.complex64)
# the location of the scatterers is randomly initialized
parameters, scat_pos = fct.position_scatterer_random(parameters)  # Initial disorder

#%% Optimize scatterers position to maximize transmission

freq = torch.tensor(7.0e9,dtype=torch.cfloat)

fct.optimize_transmission_with_pos(parameters, freq)