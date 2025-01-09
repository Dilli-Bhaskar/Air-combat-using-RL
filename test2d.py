import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch

# Load your trained DQN model
file_ = 'weights/attack_30000'
dqn = torch.load(f'{file_}.pkl')

# Set up the figure and the 2D axis
fig, ax = plt.subplots()
colorbar = None


# Predefined parameters
pusai_range = np.arange(-1*np.pi, 1*np.pi, np.pi/180)
v_range = np.arange(30, 150, 2)
z_, theta_ = 4000.00001, 0
zb_, vb_, thetab_, psib_ = 4000, 60, 0, 0

# self.state_red = [np.random.uniform(1500, 2200), np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-0.5*np.pi, 0.5*np.pi)] 
# self.state_blue = [np.random.uniform(200, 1800), np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-0.3*np.pi, 0.3*np.pi)] 



        
        
# Vectorized function to calculate V_pi_new(s)
def calculate_vpi_grid(x, y, target_x, target_y ):
    grid = np.zeros((len(x), len(y)))
    
    xb_, yb_ = target_x, target_y 



    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            states = []
            for psi_ in pusai_range:

                for v_ in v_range:
                    
                    d0 = np.sqrt((xb_ - x_) ** 2 + (yb_ - y_) ** 2 + (zb_ - z_) ** 2)
                    
                    v_r_vector = [v_ * np.cos(theta_) * np.cos(psi_), v_ * np.cos(theta_) * np.sin(psi_), v_ * np.sin(theta_)]
                    v_b_vector = [vb_ * np.cos(thetab_) * np.cos(psib_), vb_ * np.cos(thetab_) * np.sin(psib_), vb_ * np.sin(thetab_)]
                    v_r_vector_modules = np.sqrt(v_r_vector[0] ** 2 + v_r_vector[1] ** 2 + v_r_vector[2] ** 2)
                    v_b_vector_modules = np.sqrt(v_b_vector[0] ** 2 + v_b_vector[1] ** 2 + v_b_vector[2] ** 2)
                    D_RB = [xb_ - x_, yb_ - y_, zb_ - z_] # Relative position vector
                    D_BR = [x_ - xb_, y_ - yb_, z_ - zb_]
                    D_RB_modules = np.sqrt(D_RB[0] ** 2 + D_RB[1] ** 2 + D_RB[2] ** 2)
                    D_BR_modules = np.sqrt(D_BR[0] ** 2 + D_BR[1] ** 2 + D_BR[2] ** 2)
                    
                    # Angle between Relative position vector & velocity vector
                    # print("")
                    # print(D_RB, D_RB_modules)
                    
                    num = round((D_RB[0] * v_r_vector[0] + D_RB[1] * v_r_vector[1] + D_RB[2] * v_r_vector[2]), 3)
                    den = round((D_RB_modules * v_r_vector_modules), 3)
                    qr = np.arccos(num/ den) 
                    
                    
                    
                    qb = np.arccos((D_BR[0] * v_b_vector[0] + D_BR[1] * v_b_vector[1] + D_BR[2] * v_b_vector[2]) / (
                                D_BR_modules * v_b_vector_modules))
                    
                    if not -1<=(num / den)<=1:
                        print("Encountered zero norm:")
                        print("num:", num)
                        print("den:", den)
                        
                    v_x = v_r_vector[0] - v_b_vector[0]
                    v_y = v_r_vector[1] - v_b_vector[1]
                    v_z = v_r_vector[2] - v_b_vector[2]
               
                          
                    psi_D = np.arctan2(yb_ - y_, xb_ - x_) + (2 * np.pi if np.arctan2(yb_ - y_, xb_ - x_) < 0 else 0)
                    theta_D = np.arcsin((zb_ - z_)/d0)
        
        
                    state = torch.FloatTensor([(x_ - xb_)/5000,
                                                (y_ - yb_)/5000,
                                                z_/5000,
                                                d0/5000,
                                                theta_D/np.pi,
                                                psi_D/ 2*np.pi,
                                                v_x/(150-30),
                                                v_y/(150-30),
                                                v_z/(150-30), 
                                                qr/np.pi,
                                                qb/np.pi])
                    states.append(state)
            
            # Stack states and get Q values for each in a single pass through DQN
            states = torch.stack(states)
            q_values = dqn.policy_net(states)

            # Get max V_pi_new from the Q values
            max_v = torch.max(q_values).item()
            grid[i, j] = max_v

    return grid

# Plotting function that updates with sliders
def update(val):
    global colorbar  # Refer to the global colorbar
    ax.clear()

    # Generate a grid of x_, y_
    n_grids = 20
    x_ = np.linspace(500, 10000, n_grids)
    y_ = np.linspace(500, 10000, n_grids)
    
    target_x, target_y = 5000, 5000

    # Calculate the grid of V_pi_new(s)
    grid = calculate_vpi_grid(x_, y_, target_x, target_y )

    # Plot the grid values using imshow (as a heatmap)
    x_ = np.linspace(0, 10000, n_grids)
    y_ = np.linspace(0, 10000, n_grids)
    im = ax.imshow(grid.T, extent=[x_.min(), x_.max(), y_.min(), y_.max()], origin='lower', cmap='viridis', aspect='auto', interpolation='nearest')

    # Overlay the red grid
    
    grid_size = 10000 / n_grids
    ax.add_patch(plt.Rectangle((target_x - grid_size, target_y - grid_size), grid_size, grid_size, color='red', alpha=0.9))

    # Add V_pi_new values as text at each grid cell center
    for i, x in enumerate(x_):
        for j, y in enumerate(y_):
            ax.text(x, y, f'{grid[i, j]:.2f}', color='black', ha='center', va='center', fontsize=8)

    # Remove the existing colorbar if it exists
    if colorbar:
        colorbar.remove()

    # Add a new colorbar to the right
    colorbar = fig.colorbar(im, ax=ax)
    ax.legend()

    plt.title(file_)
    plt.draw()

# Initial plot
update(None)
plt.show()
