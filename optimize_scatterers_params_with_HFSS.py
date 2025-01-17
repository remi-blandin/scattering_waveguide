import torch
import numpy as np
from scipy.constants import c  # Speed of light in vacuum
import scattering_waveguide_functions as fct
import csv
import skrf as rf
import matplotlib.pyplot as plt

#%% Define functions
##############################################################################

#%% To plot and compare the matrices

def plot_compare_scat_mat(S, S_HFSS, S_diff):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    im1 = ax1.imshow(np.abs(S.numpy()))
    ax1.set_title('Dipole model')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(np.abs(S_HFSS))
    ax2.set_title('HFSS')
    plt.colorbar(im2, ax=ax2)
    
    im3 = ax3.imshow(np.abs(np.asarray(S_diff)))
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)
    
    plt.show()
    
#%% Compare matrices

def compare_matrices(S_HFSS, S):
    S_diff = torch.abs(S_HFSS) - torch.abs(S)
    loss = torch.sum(torch.abs(S_diff))
    
    return S_diff, loss

#%% Define parameters

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
    'max_nb_iterations':2000,                               # max number of optimization iterations
    'learning_rate': 0.001,                                 # learning rate for Adam optimization algorithm
    'maximal_loss': 0.1,                                   # optimization stops when loss is lower 
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

depths = np.arange(1, 9, 1)
nb_depths = len(depths)
optimized_pol = np.zeros(nb_depths, dtype=complex)

# loop over depths
for idx_d, d in enumerate(depths):

    file_name_HFSS_scat_mat = 'scat_mat_HFSS_' + str(d) + 'mm.s8p'
    network = rf.Network(file_name_HFSS_scat_mat)
    
    freqs_HFSS = network.f
    index = np.argmin(np.abs(freqs_HFSS - freq.numpy()))
    S_HFSS = network.s[index, :, :]
    
    #%% explore polarizability effect by varying its value
    
    show_matrices_at_each_iteration = False
    
    alpha_range = 5.
    nb_var = 11
    alpha_var_re = torch.arange(-alpha_range, alpha_range, alpha_range/nb_var)
    nb_alpha_re = len(alpha_var_re)
    
    alpha_var_im = torch.arange(-alpha_range, alpha_range, alpha_range/nb_var)
    nb_alpha_im = len(alpha_var_im)
    
    record = torch.zeros(nb_alpha_im, nb_alpha_re)
    
    for itre, al_re in enumerate(alpha_var_re):
        for itim, al_im in enumerate(alpha_var_im):
            
            # vary the polarizability
            parameters['alphas0'] = torch.full((nb_scat,), al_re + 1j * al_im, \
                                               dtype=torch.complex64)
            
            # compute scattering matrix
            loss,S,R = fct.calculate_loss(parameters, freq)
            
            # compute difference
            S_diff, loss = compare_matrices(torch.tensor(S_HFSS), S)
            record[itim, itre] = loss
        
            # plot the scattering matrices at the selected frequency freq
            if show_matrices_at_each_iteration:
                plot_compare_scat_mat(S, S_HFSS, S_diff)
                
    # localize the best configuration
    min_value, min_index = torch.min(record.flatten(), 0)
    min_index_2d = divmod(min_index.item(), record.size(1))
    best_alpha = alpha_var_re[min_index_2d[1]] + 1j * alpha_var_im[min_index_2d[0]]
    print(f'Best alpha = {torch.real(best_alpha).item():.4f} \
          + j {torch.imag(best_alpha).item():.4f} Loss: {loss:.2f}')
                
    # plot difference with respect to polarizability
    x_min = alpha_var_re.min().item()
    x_max = alpha_var_re.max().item()
    y_min = alpha_var_im.min().item()
    y_max = alpha_var_im.max().item()
    plt.imshow(np.log(record.detach().numpy()), extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.plot(torch.real(best_alpha).item(), torch.imag(best_alpha).item(), 'o:r')
    plt.show()
                
    # compute the scattering matrix of the best alpha
    parameters['alphas0'] = torch.full((nb_scat,), best_alpha, dtype=torch.complex64)
    loss,S,R = fct.calculate_loss(parameters, freq)
    
    # plot scattering matrix of the best alpha
    S_diff, loss = compare_matrices(torch.tensor(S_HFSS), S)
    plot_compare_scat_mat(S, S_HFSS, S_diff)
        
    #%% Optimize polarizability
    
    # pol_optim = torch.tensor([0.1, 5.])
    pol_optim = torch.tensor([np.real(best_alpha), np.imag(best_alpha)])
    pol_optim.requires_grad_(True)
    
    optimizer = torch.optim.Adam([pol_optim], betas=(0.99, 0.999), \
                                 lr = parameters['learning_rate'], \
                                     amsgrad=True, eps=1e-8)
    
    record_opt = np.zeros(parameters['max_nb_iterations'])
        
    iteration = 0
    while iteration < parameters['max_nb_iterations']  and loss.item() > parameters['maximal_loss'] :
        
        optimizer.zero_grad(set_to_none = True)
        
        # update polarizability value
        parameters['alphas0'] = pol_optim[0].repeat(nb_scat) + 1j * pol_optim[1].repeat(nb_scat)
        loss_old,S,R = fct.calculate_loss(parameters, freq)
        S_diff, loss = compare_matrices(torch.tensor(S_HFSS), S)
        record_opt[iteration] = loss
        loss.backward()     # Backpropagate the loss
        optimizer.step()    # Step the optimizer
        
        parameters['alphas0'].detach_()
        
        if iteration % 100 == 0:
            print(f'Iteration {iteration}: Loss = {loss.item()} Pol = \
                    {pol_optim[0].item():.2f} + j {pol_optim[1].item():.2f}')
            fct.plot_optimization_progress(parameters, record_opt[:iteration], S.detach())
        
        iteration = iteration + 1
        
    # plot scattering matrix of the best alpha
    S_diff, loss = compare_matrices(torch.tensor(S_HFSS), S)
    S.detach_()
    S_diff.detach_()
    plot_compare_scat_mat(S, S_HFSS, S_diff)
    
    # save the optimized polarizability
    optimized_pol[idx_d] = pol_optim[0].item() + 1j * pol_optim[1].item()

#%% Plot the optimized polarizations

plt.plot(depths, np.real(optimized_pol), label='Real part')
plt.plot(depths, np.imag(optimized_pol), label='Imag part')
plt.xlabel('Depth (mm)')
plt.ylabel('Polarizabiity')
plt.legend()
plt.show()