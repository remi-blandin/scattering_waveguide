import torch
import numpy as np
from scipy.constants import c  # Speed of light in vacuum
import scattering_waveguide_functions as fct

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
    'max_nb_iterations':100,                               # max number of optimization iterations
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
alphas_teflon = +1j * 0.05
alphas_optim = +1j * 1.

#%% generate scatterers polarization and position

# number and type of scatterers
nb_opt_metal = 12       # Metal scatterers to optimize
nb_opt_dielec = 10      # Teflon scatterers to optimize
nb_scat = nb_opt_metal + nb_opt_dielec  # Number of dipoles to be optimized (the nb_opt at the left part)
n_optim = 10

# radius of the different scatterers types
radius = 0.0021

# Generate the complex medium through which transmission is optimized
#############################################################################

# the metal scatterers are placed at random positions in the polarization vector
index_metal = torch.randperm(nb_scat)[:nb_opt_metal]
# first the polarization vector is initialized with the teflon property
parameters['alphas0'] = torch.full((nb_scat,), alphas_teflon, dtype=torch.complex64)
# then the indexes corresponding to metal are changed to the metal property
parameters['alphas0'][index_metal] = torch.full((nb_opt_metal,), alphas_metal, dtype=torch.complex64)
# the location of the scatterers is randomly initialized
parameters['scatRad'] = torch.full((nb_scat,), radius)
# put scatterers at random positions
parameters, scat_pos = fct.position_scatterer_random(parameters, \
                        x_min=parameters['H_guide']/2, show_progress=True)

# Generate the complex medium whose polarizability is optimized
#############################################################################

parameters['alphas0'] = torch.cat((parameters['alphas0'], \
                                   torch.full((n_optim, ), alphas_optim)))
    
parameters['scatRad'] = torch.cat((parameters['scatRad'], \
                                       torch.full((n_optim,), radius)))

parameters, scat_pos = fct.position_scatterer_random(parameters, \
                                x_max=parameters['H_guide']/2, nb_scat=n_optim, show_progress=True)
    
fct.plot_scatterers_config(parameters)


#%% Optimize scatterers polarizability to maximize transmission

freq = torch.tensor(7.0e9,dtype=torch.cfloat)

# get indexes of dielectric scatterers
is_dielectric = parameters['alphas0'] == alphas_optim
idx_diel_scat = is_dielectric.nonzero()

# to store the progress of the optimisation later
# nb_repeat = 2
nb_diel_scat = len(idx_diel_scat)
record = np.zeros(parameters['max_nb_iterations'])

loss,S,R = fct.calculate_loss(parameters, freq)
loss_ref = loss
print(f'Loss = {loss.item()} Reflection = {R.item()}')
fct.plot_optimization_progress(parameters, record, S.detach())

# pol_optim = np.imag(parameters['alphas0'][idx_diel_scat])
pol_optim = torch.full((nb_diel_scat, 1), 1., dtype=torch.float32, requires_grad=True)
# pol_optim.requires_grad_(True)

optimizer = torch.optim.Adam([pol_optim], betas=(0.99, 0.999), \
                             lr = parameters['learning_rate'], \
                                 amsgrad=True, eps=1e-8)

iteration = 0
while iteration < parameters['max_nb_iterations']  and loss.item() > parameters['maximal_loss'] :
    
    optimizer.zero_grad(set_to_none = True)
    
    # update scatterers polarizability in parameters
    parameters['alphas0'][idx_diel_scat] = 1j * pol_optim
    
    loss,S,R = fct.calculate_loss(parameters, freq)
    record[iteration] = loss
    loss.backward()     # Backpropagate the loss
    optimizer.step()    # Step the optimizer
    
    # with torch.no_grad():
    #     pol_optim = pol_optim + 0.001 * (torch.rand(len(idx_diel_scat),1) - 0.5)
    
    iteration = iteration + 1
    
    print(f'Iteration {iteration}: Loss = {loss.item()} Reflection = {R.item()}')
    fct.plot_optimization_progress(parameters, record[:iteration], S.detach())
    
# for ii in range(0,nb_repeat):
#     for it, idx in enumerate(idx_diel_scat):
#         # switch the polarizability 
#         parameters['alphas0'][idx] = -parameters['alphas0'][idx]
        
#         loss,S,R = fct.calculate_loss(parameters, freq)
#         print(f'Iteration {it}: Loss = {loss.item()} Reflection = {R.item()}')
#         record[it + ii*nb_diel_scat] = loss
        
#         fct.plot_optimization_progress(parameters, record[:it + 1 + ii*nb_diel_scat], S.detach())
        
#         if loss > loss_ref:
#             # switch back to the original configuration
#             parameters['alphas0'][idx] = -parameters['alphas0'][idx]
#         else:
#             loss_ref = loss
            
#     # fct.plot_optimization_progress(parameters, record[:it + ii*nb_diel_scat], S.detach())
    
# # Show best config
# loss,S,R = fct.calculate_loss(parameters, freq)
# print(f'Loss = {loss.item()} Reflection = {R.item()}')
# fct.plot_optimization_progress(parameters, record, S.detach())