import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.constants import c  # Speed of light in vacuum

#%% position scatterers on a line

def generate_scatterers_line(parameters, coord_start, coord_end, const_coord, aligned, \
                             nb_scat, polarization, radius):
    
    # generate the coordinate whih remains constant
    const_coord = torch.full((nb_scat, ), const_coord)
    
    # generate the varying coordinate
    coord = coord_start + torch.range(1, nb_scat) * (coord_end - coord_start) / (nb_scat + 1)
    
    # put the coordinates in the parameters according to the configuration: 
    #   - aligned with the waveguide axis
    #   - or perpendicular to the waveguide axis
    if aligned:
        parameters['posx'] = torch.cat((parameters['posx'], coord))
        parameters['posy'] = torch.cat((parameters['posy'], const_coord))
    else: # perpendicular
        parameters['posx'] = torch.cat((parameters['posx'], const_coord))
        parameters['posy'] = torch.cat((parameters['posy'], coord))
        
    # put the polarization of the scatterers in the parameters
    parameters['alphas0'] = torch.cat((parameters['alphas0'], \
                                       torch.full((nb_scat, ), polarization)))
        
    # put the radius of the scatterers in the parameters
    parameters['scatRad'] = torch.cat((parameters['scatRad'], \
                                       torch.full((nb_scat,), radius)))
        
    # update the number of scatterers
    parameters['nb_scat'] = len(parameters['alphas0'])
        


#%% randomly attribute location to the scatterers

def position_scatterer_random(parameters, x_min=0, x_max=0, nb_scat=0, show_progress=False):

    if nb_scat == 0:
        nb_scat = len(parameters['alphas0'])
        idx_start = 0
        parameters['posx'] = torch.empty(nb_scat)
        parameters['posy'] = torch.empty(nb_scat)
    else:
        idx_start = len(parameters['posx'])
        parameters['posx'] = torch.cat((parameters['posx'], torch.empty(nb_scat)))
        parameters['posy'] = torch.cat((parameters['posy'], torch.empty(nb_scat)))
        
    max_scatterers_rad = parameters['scatRad'][0].item() * 2
    
    dmin_wall = parameters['spacing_min']   # minimal distance with the walls
    
    # range in which the coordinates are generated
    x_min = max(x_min, dmin_wall)
    x_min = min(x_min, parameters['H_guide'] - dmin_wall)
    if x_max == 0:
        x_max = parameters['H_guide'] - dmin_wall
    else:
        x_max = max(x_max, dmin_wall)
        x_max = min(x_max, parameters['H_guide'] - dmin_wall)
    y_min = dmin_wall
    y_max = parameters['W_guide'] - dmin_wall

    scatterers_pos_wrong = True
    while scatterers_pos_wrong:

        # randomly generate scaterrers position
        parameters['posx'][idx_start : idx_start + nb_scat] = \
                                       x_min + (x_max - x_min) * np.random.rand(nb_scat)
        parameters['posy'][idx_start : idx_start + nb_scat] = \
                                       y_min + (y_max - y_min) * np.random.rand(nb_scat)

        ensure_no_overlap(parameters, max_scatterers_rad)

        scatterers_pos_wrong = condition_scatterers(parameters)
        
        if show_progress:
            plot_scatterers_config(parameters)

    # sort the coordinates by increasing x
    posx = parameters['posx'][idx_start : idx_start + nb_scat] 
    posy = parameters['posy'][idx_start : idx_start + nb_scat] 
    sorted_indices = np.argsort(posx)
    posx = posx[sorted_indices]
    posy = posy[sorted_indices]
    parameters['posx'][idx_start : idx_start + nb_scat] = posx
    parameters['posy'][idx_start : idx_start + nb_scat] = posy
    
    # sorted_indices = np.argsort(parameters['posx'])
    # parameters['posx'] = parameters['posx'][sorted_indices]
    # parameters['posy'] = parameters['posy'][sorted_indices]

    dippos = torch.zeros((nb_scat, 2))
    dippos[:nb_scat, 0] = parameters['posy'][:nb_scat].clone().detach() # Random position of the dipoles in y
    dippos[:nb_scat, 1] = parameters['posx'][:nb_scat].clone().detach() # Random position of the dipoles in x
    
    # update the number of scatterers
    parameters['nb_scat'] = len(parameters['alphas0'])

    return parameters, dippos
    
#%% Avoid scatterers to overlap

def ensure_no_overlap(parameters,radius):
    # Ensure the dipoles are not overlapping
    dippos = torch.zeros((len(parameters['posx']), 2))
    # dippos = torch.zeros((parameters['nb_scat'], 2))
    dippos[:,0] = parameters['posy'].clone()
    dippos[:,1] = parameters['posx'].clone()
    for i in range(dippos.size(0)):
        for j in range(i + 1, dippos.size(0)):
            dist = torch.sqrt((dippos[i, 0] - dippos[j, 0])**2 + (dippos[i, 1] - dippos[j, 1])**2)
            if dist < 2 * radius:
                overlap = 2 * radius - dist
                direction = (dippos[j] - dippos[i]) / dist
                dippos[i] -= overlap / 2 * direction *1.001
                dippos[j] += overlap / 2 * direction *1.001
    parameters['posy'] = dippos[:,0] 
    parameters['posx'] = dippos[:,1] 
    return parameters

#%% Assess that the scatterers's location satisfy the geometrical constraints:
    # minimum distance between scatterers and from the walls

def condition_scatterers(parameters):
    # Extract parameters
    posx = parameters['posx'].clone().detach()
    posy = parameters['posy'].clone().detach()
    scatRad = parameters['scatRad'].clone().detach()
    W_guide = parameters['W_guide']
    H_guide = parameters['H_guide']
    n_scat = len(posx)

    dmin = 0.0001
    dmin_wall = parameters['spacing_min']

    # Initialize
    a = []

    ## THE FOLLOWING 4 CHECKS COULD BE AVOIDED BY INCLUDING THESE BOUNDARIES 
    ## IN THE COORDINATES RANDOM GENERATION
    # Scatterers not touching the low boundary
    for rad in scatRad:
        a.append((posy <= rad + dmin_wall).sum().item())

    # Scatterers not touching the upper boundary
    for rad in scatRad:
        a.append((posy > W_guide - rad - dmin_wall).sum().item())

    # Scatterers not touching the left boundary
    for rad in scatRad:
        a.append((posx < rad + dmin_wall).sum().item())

    # Scatterers not touching the right boundary
    for rad in scatRad:
        a.append((posx > H_guide - rad - dmin_wall).sum().item())

    # Distance between scatterers
    d = torch.zeros((n_scat, n_scat))

    for ii in range(n_scat):
        d[:, ii] = torch.sqrt((posx - posx[ii]) ** 2 + (posy - posy[ii]) ** 2)

    for ii in range(n_scat - 2):
        for jj in range(ii + 1, n_scat - 1):
            if d[ii, jj] < scatRad[ii] + scatRad[jj] + dmin:
                a.append(1)

    # Condition
    scatterers_pos_wrong = (sum(a) != 0)
    return scatterers_pos_wrong

def distance_dippos(dippos):
    # Distance between scatterers
    n_scat = dippos.shape[0]
    d = torch.zeros((n_scat, n_scat))

    for ii in range(n_scat):
        d[:, ii] = torch.sqrt((dippos[:,0] - dippos[ii,0]) ** 2 + (dippos[:,1] - dippos[ii,1]) ** 2)

    return d

#%% Evaluate the loss function

def calculate_loss(parameters, freq):
    
    n_port = parameters['Nport']
    
    S = scattering_matrix(parameters, freq) 
    Tx = S[n_port:2*n_port, :n_port]
    U, lambda1, v = torch.linalg.svd(Tx)
    R = 1-torch.mean(torch.sum(torch.abs(Tx)**2,axis=1))
    # loss = (1-lambda1[parameters['Nport']-1] )+R*(1-R)
    loss = R
    
    return loss, S, R

#%% Compute the scattering matrix

def scattering_matrix(parameters, freq):
    
    nb_scat = parameters['nb_scat']
    alphas0 = parameters['alphas0']
    N = parameters['Nport'].clone()
    n_s = parameters.get('n_s', 1) # number of dipoles to model one scatterer
    
    # Computation if data from previous computation are saved
    ###########################################################################
    
    if parameters['use_precomputed_values'] and 'S0' in parameters:
        
        S_scat = 0
        if nb_scat > 0:
            
            # get the saved parameters
            G = parameters['G']
            S0 = parameters['S0']
            dipole_pos = parameters['dipole_pos']

            # duplicate polarizabilities
            if n_s > 1:
                alphas0 = alphas0.repeat(n_s)  
            
            # compute the scattering matrix of the scatterers
            mat = torch.diag(1 / alphas0)
            green = Green_function(dipole_pos, dipole_pos, parameters, freq)
            mat = mat - green
            W1 = torch.pinverse(mat)
            
            S_scat = torch.mm(torch.mm(G.t(), W1), G)
            
            # Compute the total scattering matrix
            
            S = S0 + S_scat
            S[:N, :N] = -S[:N, :N]  
            S[N:, N:] = -S[N:, N:]  
            
    else:
        
        # Full computation
        ###########################################################################
   
        W = parameters['W_guide'].clone()
        L = parameters['H_guide'].clone()
    
        if nb_scat > 0:
    
            # if the scatterers are represented by multiple dipoles, the dipoles
            # corresponding to each scatterer are generated
            if n_s > 1:
                theta = torch.linspace(0, 2 * np.pi, n_s + 1)
                dipole_pos = torch.zeros((n_s * nb_scat, 2))
                for ii in range(nb_scat):
                    R = parameters['scatRad'][ii]  # Radius of scatterers
                    dipole_pos[n_s * ii:n_s * (ii + 1), 0] = parameters['posy'][ii] + R * torch.cos(theta[:-1])
                    dipole_pos[n_s * ii:n_s * (ii + 1), 1] = parameters['posx'][ii] + R * torch.sin(theta[:-1])
                alphas0 = alphas0.repeat(n_s)  # also duplicate polarizabilities
                
            parameters['dipole_pos'] = dipole_pos
    
        # waveguide modes parameters
        nu_c = c / (2 * W)
        kn = (2 * np.pi / c) * torch.sqrt(freq**2 - (torch.arange(1, 301) * nu_c)**2) 
    
        # Calculate the green function from the input to the dipoles
        G2 = torch.sqrt(2 / W) * torch.sin(dipole_pos[:, 0].unsqueeze(1) * torch.arange(1, N + 1) * np.pi / W) * \
             torch.exp(-1j * dipole_pos[:, 1].unsqueeze(1) * torch.conj(kn[:N]))
        G2 = -torch.pow(-1.0, torch.arange(1, N + 1)) * G2
        G2 = (1 / torch.sqrt(kn[:N])) * G2
    
        # calculate the Green functions from the dipoles to the output 
        G1 = torch.sqrt(2 / W) * torch.sin(dipole_pos[:, 0].unsqueeze(1) * torch.arange(1, N + 1) * np.pi / W) * \
             torch.exp(-1j * torch.abs(L - dipole_pos[:, 1].unsqueeze(1)) * torch.conj(kn[:N]))
        G1 = -torch.pow(-1.0, torch.arange(1, N + 1)) * G1
        G1 = (1 / torch.sqrt(kn[:N])) * G1
    
        # concatenate the Green functions of the scatterers
        G = torch.cat((G2, G1), dim=1)
        
        # save G for later computations
        parameters['G'] = G
    
        # compute the scattering matrix of the scatterers
        S_scat = 0
        if nb_scat > 0:
            mat = torch.diag(1 / alphas0)
            green = Green_function(dipole_pos, dipole_pos, parameters, freq)
            mat = mat - green
            W1 = torch.pinverse(mat)
            S_scat = torch.mm(torch.mm(G.t(), W1), G)
    
        # Compute the scattering matrix of the waveguide without scatterers
        S0 = torch.zeros((2 * N, 2 * N), dtype=torch.complex128)
        G0 = torch.exp(-1j * L * torch.conj(kn[:N]))
        t0 = torch.diag(G0)
        S0[:N, N:2 * N] = t0
        S0[N:2 * N, :N] = t0
        S0 = -S0
        
        # save S0 for later computations 
        parameters['S0'] = S0
    
        # Compute the total scattering matrix
        S = S0 + S_scat
        S[:N, :N] = -S[:N, :N]  
        S[N:, N:] = -S[N:, N:]  

    return S

#%% Compute the Green function

def Green_function(pos1, pos2, parameters, frequency):

    # Function parameters
    W_guide = parameters['W_guide']
    N_mode_use = parameters['N_mode_use']

    # Derived parameters
    nu_c = c / (2 * W_guide)

    # Calculate wavenumbers
    n = torch.arange(1, 101, dtype=torch.float64)
    kn = (2 * torch.pi / c) * torch.sqrt(frequency ** 2 - (n * nu_c) ** 2)

    # Positions
    x1, y1 = pos1[:, 0], pos1[:, 1]
    x2, y2 = pos2[:, 0], pos2[:, 1]

    # Initialize Green's function matrix
    G = torch.zeros((pos1.shape[0], pos2.shape[0]), dtype=torch.cfloat)

    # Vectorized computation for all modes
    for nn in range(1, N_mode_use + 1):
        kn_n = torch.conj(kn[nn - 1])
        sin_term1 = torch.sin(nn * torch.pi * x1.unsqueeze(1) / W_guide)
        sin_term2 = torch.sin(nn * torch.pi * x2.unsqueeze(0) / W_guide)
        exp_term = torch.exp(-1j * kn_n * torch.abs(y1.unsqueeze(1) - y2.unsqueeze(0)))
        
        A = (-sin_term1 * sin_term2 * 2 / W_guide / kn_n) * exp_term
        G += A

    return G

#%% Plot the position and nature of the scatterers

def plot_scatterers(parameters, ax):
    
    ax.set_aspect('equal')
    
    scat_color = np.round(np.imag(parameters['alphas0'].detach()) * 10.)
    scat_color = scat_color - min(scat_color)
    scat_color =  scat_color/max(scat_color)
    
    n_scatt = len(parameters['posx'])
    for s in range(0, n_scatt):
        circ = plt.Circle((parameters['posx'][s], parameters['posy'][s]),  \
                          parameters['scatRad'][s], color=mcolors.to_rgb((scat_color[s].item(), 0, 0)))
        ax.add_patch(circ)
        
    # draw the waveguide contours 
    rect = plt.Rectangle((0., 0.), parameters['H_guide'], parameters['W_guide'],\
                             linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
        
    x_min = min(0., min(parameters['posx']))
    x_max = max(parameters['H_guide'], max(parameters['posx']))
    ax.set_xlim((x_min, x_max))
    
    y_min = min(0., min(parameters['posy']))
    y_max = max(parameters['W_guide'], max(parameters['posy']))
    ax.set_ylim((y_min, y_max))
    
    # print(f'x_min = {x_min} x_max = {x_max} y_min = {y_min} y_max = {y_max}')
    
def plot_scatterers_config(parameters):
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    plot_scatterers(parameters, ax)
    plt.show()
    
#%% Plot the progress of the optimization

def plot_optimization_progress(parameters, record, S):
    
    fig = plt.figure()
    
    # plot scatterers position and nature
    ax1 = plt.subplot(2, 1, 1)
    plot_scatterers(parameters, ax1)
    
    # plot the evolution of the loss over the iterations
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(record)
    
    # plot the scattering matrix
    ax3 = plt.subplot(2, 2, 4)
    im = ax3.imshow(torch.abs(S)**2, cmap='viridis', vmin=0, vmax=1)
    im.set_clim(0, 1)
    fig.colorbar(im)
    
    plt.show()

#%% Optimize trqnsmission by chqnging the position of the scqtterers

def optimize_transmission_with_pos(parameters, freq):
    
    loss,S,R = calculate_loss(parameters, freq)
    print(f'Loss = {loss.item()} Reflection = {R.item()}')
    
    # to store the progress of the optimisation later
    record = np.zeros(parameters['max_nb_iterations'])
    
    # plot scattering matrix
    plot_optimization_progress(parameters, record, S.detach())
    
    #%% Optimize scatterers position to maximize transmission
    
    min_dist_scat = parameters['scatRad'][0].item() * 4
    
    scat_pos = torch.cat( (parameters['posy'].unsqueeze(1), \
                           parameters['posx'].unsqueeze(1)), dim=1) 
    scat_pos.requires_grad_(True)
    
    optimizer = torch.optim.Adam([scat_pos], betas=(0.99, 0.999), \
                                 lr = parameters['learning_rate'], \
                                     amsgrad=True, eps=1e-8)
    
    iteration = 0
    while iteration < parameters['max_nb_iterations']  and loss.item() > parameters['maximal_loss'] :
        
        optimizer.zero_grad(set_to_none = True)
        
        # update scatterers position in parameters
        parameters['posx'] = scat_pos[:,1]
        parameters['posy'] = scat_pos[:,0]
        
        loss,S,R = calculate_loss(parameters, freq)
        record[iteration] = loss
        loss.backward()     # Backpropagate the loss
        optimizer.step()    # Step the optimizer
        
        # Ensure the dipoles are not touching the boundaries
        with torch.no_grad():
            
            scat_pos[:,0].clamp_(min=torch.max(parameters['scatRad'])*1.1)
            scat_pos[:,0].clamp_(max=parameters['W_guide']-torch.max(parameters['scatRad'])*2)
            scat_pos[:,1].clamp_(min=torch.max(parameters['scatRad'])*1.1)
            scat_pos[:,1].clamp_(max=parameters['H_guide']-torch.max(parameters['scatRad'])*2)
            
            ensure_no_overlap(parameters,  min_dist_scat)
            
        if iteration % 20 == 0:
            print(f'Iteration {iteration}: Loss = {loss.item()} Reflection = {R.item()}')
            plot_optimization_progress(parameters, record[:iteration], S.detach())
            
        iteration = iteration + 1
            
    print(f'Iteration {iteration}: Loss = {loss.item()} Reflection = {R.item()}')
    plot_optimization_progress(parameters, record[:iteration], S.detach())