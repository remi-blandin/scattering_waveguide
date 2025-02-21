import torch
import numpy as np
from scipy.constants import c  # Speed of light in vacuum
import scattering_waveguide_functions as fct
import os
import random

parameters = {
    'W_guide': torch.tensor(0.1),                           # Waveguide width
    'H_guide': torch.tensor(0.4),                           # Waveguide length
    'epsr': torch.tensor(2.1),                              # Relative permittivity
    'frequency': torch.tensor(7.0e9,dtype=torch.cfloat),                         # Frequency (can be a vector)
    'spacing_min': torch.tensor(0.001),                     # Minimal distance to the middle
    'N_mode_use': 23,                                       # Total number of modes taken into account in the calculation of the Green function
    'n_s': torch.tensor(8),                                 # number of dipoles to model one cylinder
    'method': 'random',                                     # Optimization method (random in this case)
    'max_nb_iterations':1000,                               # max number of optimization iterations
    'learning_rate': 0.001,                                 # learning rate for Adam optimization algorithm
    'maximal_loss': 0.01,                                   # optimization stops when loss is lower 
    'use_precomputed_values': False
    }

load_saved_config = True
file_name_rand_config = 'random_config.pt'

initialize_randomly_polarizabilities = True

# polarizabilities
alphas_metal = -1j * 6
alphas_teflon = +1j * 0.05
alphas_side = -1j * 5.
    
# compute the number of propagating modes
nu_c = c / (2 * parameters['W_guide'])  # Cut-off frequencies
kn = (2 * np.pi / c) * torch.rsqrt(parameters['frequency']**2 - (torch.arange(1, 1001) * nu_c)**2)  # Waveguide mode k parameters
parameters['Nport'] = torch.max(torch.nonzero(torch.imag(kn) == 0)) + 1  # Number of modes of the empty waveguide (S is a matrix of dimension 2Nx2N)
print(f"Number of propagating modes: {parameters['Nport']}")

#%% generate scatterers polarization and position

# number and type of scatterers
nb_opt_metal = 22       # Metal scatterers to optimize
nb_opt_dielec = 18      # Teflon scatterers to optimize
nb_scat = nb_opt_metal + nb_opt_dielec  # Number of dipoles to be optimized (the nb_opt at the left part)
n_bars = 3
n_side_scat = 15

# distance between the walls and the scatteres lines
wall_dist = 0.1 * parameters['W_guide']

# radius of the different scatterers types
rad_teflon = 0.0021
rad_scat = 0.002
rad_bars = 0.0031

# Generate the complex medium through which transmission is optimized
#############################################################################

if load_saved_config and os.path.exists(file_name_rand_config):
    
    svaed_config = torch.load(file_name_rand_config)
    parameters['alphas0'] = svaed_config['alphas0']
    parameters['scatRad'] = svaed_config['scatRad']
    parameters['posx'] = svaed_config['posx']
    parameters['posy'] = svaed_config['posy']
    
    parameters['nb_scat'] = len(parameters['alphas0'])
    
else:
    
    # the metal scatterers are placed at random positions in the polarization vector
    index_metal = torch.randperm(nb_scat)[:nb_opt_metal]
    # first the polarization vector is initialized with the teflon property
    parameters['alphas0'] = torch.full((nb_scat,), alphas_teflon, dtype=torch.complex64)
    # then the indexes corresponding to metal are changed to the metal property
    parameters['alphas0'][index_metal] = torch.full((nb_opt_metal,), alphas_metal, dtype=torch.complex64)
    # the location of the scatterers is randomly initialized
    parameters['scatRad'] = torch.full((nb_scat,), rad_teflon)
    # put scatterers at random positions
    parameters, scat_pos = fct.position_scatterer_random(parameters, x_min=parameters['H_guide']/2)  # Initial disorder
    
    # save the random config
    config_to_save = {
        'alphas0': parameters['alphas0'] ,
        'scatRad': parameters['scatRad'],
        'posx': parameters['posx'],
        'posy': parameters['posy'],
        }
    torch.save(config_to_save, file_name_rand_config)

# Calculate the scattering matrix of the disordered medium without the sides
#############################################################################

freq = torch.tensor(7.0e9,dtype=torch.cfloat)

loss,S,R = fct.calculate_loss(parameters, freq)
loss_ref = loss
print(f'Loss = {loss.item()} Reflection = {R.item()}')
fct.plot_optimization_progress(parameters, np.array([loss, loss]), S.detach())

# Generate the arrays of scatterers placed before the complex medium
#############################################################################

# Generate the metal grid at the entrance
fct.generate_scatterers_line(parameters, 0, parameters['W_guide'], \
                             rad_bars, False, \
                                 n_bars, alphas_metal, rad_bars)

# Generate the scatterers at the bottom of the first half of the waveguide
fct.generate_scatterers_line(parameters, 0, parameters['H_guide'] / 2., \
                             wall_dist, True, \
                                 n_side_scat, alphas_side, rad_scat)
    
# Generate the scatterers at the top of the first half of the waveguide
fct.generate_scatterers_line(parameters, 0, parameters['H_guide'] / 2., \
                             parameters['W_guide'] - wall_dist, True, \
                                 n_side_scat, alphas_side, rad_scat)
    
fct.plot_scatterers_config(parameters)


#%% Optimize scatterers polarizability to maximize transmission

# get indexes of dielectric scatterers
is_dielectric = parameters['alphas0'] == alphas_side
idx_diel_scat = is_dielectric.nonzero()

# randomly initialize the scatterers
if initialize_randomly_polarizabilities:
    loss = 1.
    nb_it_max = 50
    it_idx = 0
    while (loss > 0.8) and (it_idx < nb_it_max):
        for it, idx in enumerate(idx_diel_scat):
            parameters['alphas0'][idx] = random.choice([1, -1]) * parameters['alphas0'][idx]
        loss,S,R = fct.calculate_loss(parameters, freq)
        print(f'Itteration {it_idx} Loss = {loss.item()} Reflection = {R.item()}')
        it_idx = it_idx + 1

# to store the progress of the optimisation later
nb_repeat = 10
nb_diel_scat = len(idx_diel_scat)
record = np.zeros(nb_repeat * nb_diel_scat)

loss,S,R = fct.calculate_loss(parameters, freq)
loss_ref = loss
print(f'Loss = {loss.item()} Reflection = {R.item()}')
fct.plot_optimization_progress(parameters, np.array([loss, loss]), S.detach())

for ii in range(0,nb_repeat):
    no_improvement = True
    for it, idx in enumerate(idx_diel_scat):
        # switch the polarizability 
        parameters['alphas0'][idx] = -parameters['alphas0'][idx]
        
        loss,S,R = fct.calculate_loss(parameters, freq)
        print(f'Iteration {it}: Loss = {loss.item()} Reflection = {R.item()}')
        record[it + ii*nb_diel_scat] = loss
        
        # fct.plot_optimization_progress(parameters, record[:it + 1 + ii*nb_diel_scat], S.detach())
        
        if loss > loss_ref:
            # switch back to the original configuration
            parameters['alphas0'][idx] = -parameters['alphas0'][idx]
        else:
            no_improvement = False
            loss_ref = loss
    if no_improvement:
        fct.plot_optimization_progress(parameters, record[:it + ii*nb_diel_scat], S.detach())
        break
            
    fct.plot_optimization_progress(parameters, record[:it + ii*nb_diel_scat], S.detach())
    
# Show best config
loss,S,R = fct.calculate_loss(parameters, freq)
print(f'Loss = {loss.item()} Reflection = {R.item()}')
fct.plot_optimization_progress(parameters,  np.array([loss, loss]), S.detach())
