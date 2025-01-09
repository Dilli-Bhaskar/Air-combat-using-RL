import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import os
import torch
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import CubicSpline
import os

np.random.seed(1)
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  #To prevent hash randomization, make the experiment reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

class Uav_Env(object):
    def __init__(self):
        self.v_min = 30
        self.v_max = 150
        
        self.h_min = 500
        self.h_max = 5000
        
        self.D_th = 5000
        self.H_th = 5000
        
        self.d_max = 1300
        self.h_best = 5000
        self.delh_best = 1000
        self.v_best = 0
        self.g = 9.81
        self.w1 = 0.5
        self.w2 = 0.25
        self.w3 = 0.25
        self.r=[]

    

        self.attk_max = np.radians(30)
        self.esp_max = np.radians(60)
        
        self.max_step = 200

        self.distance = None
        self.height = None
        self.disengagement_angle = None
        self.deviation_angle = None
        self.done = False
        
        self.fig = None
        self.ax = None
        self.red_line = None
        self.blue_line = None
        self.connection_line = None
        self.vector_quiver_blue = None
        self.vector_quiver_red = None
        self.status = None
        
        self._create_records = False
        self.dt = 1/60
        self.down_sample_step = 3
        self.internal_step = 0
        
    def derivatives(self, state, Nx, Nz, roll):
        v, theta, psi, x, y, z = state
        

        # Calculate the rate of change (derivatives) of each state variable
        dv = self.g * (Nx - np.sin(theta))
        dtheta = (self.g / v) * (Nz * np.cos(roll) - np.cos(theta))
        dpsi = (self.g * Nz * np.sin(roll)) / (v * np.cos(theta))
        dx = v * np.cos(theta) * np.cos(psi)
        dy = v * np.cos(theta) * np.sin(psi)
        dz = v * np.sin(theta)
        
        return np.array([dv, dtheta, dpsi, dx, dy, dz])
    
    def step(self, action1):
        self.internal_step += 1

        x, y, z, v, theta, psi = self.state_red
        xb, yb, zb, vb, thetab, psib = self.state_blue


        action_map = {
        0: (0, 1, 0),                        # Forward, maintain
        1: (2, 1, 0),                        # Forward, accelerate
        2: (-1, 1, 0),                       # Forward, decelerate
        3: (0, 3.5, 0),                      # Upward, maintain
        4: (2, 3.5, 0),                      # Upward, accelerate
        5: (-1, 3.5, 0),                     # Upward, decelerate
        6: (0, -3.5, 0),                     # Downward, maintain
        7: (2, -3.5, 0),                     # Downward, accelerate
        8: (-1, -3.5, 0),                    # Downward, decelerate
        9: (0, 3.5, np.arccos(2 / 7)),       # Left turn, maintain
        10: (2, 3.5, np.arccos(2 / 7)),      # Left turn, accelerate
        11: (-1, 3.5, np.arccos(2 / 7)),     # Left turn, decelerate
        12: (0, 3.5, -np.arccos(2 / 7)),     # Right turn, maintain
        13: (2, 3.5, -np.arccos(2 / 7)),     # Right turn, accelerate
        14: (-1, 3.5, -np.arccos(2 / 7)),    # Right turn, decelerate
                    }

        Nx, Nz, roll = action_map.get(action1, (0, 0, 0))  # Default to (0, 0, 0) if action1 is not in map

        Nz = 1 / np.cos(roll)  
    
        h=1
        k1 = self.derivatives(np.array([v, theta, psi, x, y, z]), Nx, Nz, roll)
        k2 = self.derivatives(np.array([v, theta, psi, x, y, z]) + 0.5 * h * k1, Nx, Nz, roll)
        k3 = self.derivatives(np.array([v, theta, psi, x, y, z]) + 0.5 * h * k2, Nx, Nz, roll)
        k4 = self.derivatives(np.array([v, theta, psi, x, y, z]) + h * k3, Nx, Nz, roll)

        new_state = np.array([v, theta, psi, x, y, z]) + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

        v_, theta_, psi_, x_, y_, z_ = new_state

        if z !=  4000.0: raise ValueError(f"Error: z is {z}, but it must be 4000.")
        
        vb_ = vb
        xb_ = xb #+ vb * np.cos(thetab) * np.cos(psib) * h
        yb_ = yb#+ vb * np.cos(thetab) * np.sin(psib) * h
        zb_ = zb #+ vb * np.sin(thetab) * h
        psib_ = psib
        thetab_ = thetab

        self.state_red = [x_,y_,z_,v_,theta_,psi_]
        self.state_blue = [xb_, yb_, zb_, vb_, thetab_, psib_]



        d0 = np.sqrt((xb_ - x_) ** 2 + (yb_ - y_) ** 2 + (zb_ - z_) ** 2)
        self.v_r_vector = [v_ * np.cos(theta_) * np.cos(psi_), v_ * np.cos(theta_) * np.sin(psi_), v_ * np.sin(theta_)]
        self.v_b_vector = [vb_ * np.cos(thetab_) * np.cos(psib_), vb_ * np.cos(thetab_) * np.sin(psib_), vb_ * np.sin(thetab_)]
        self.v_r_vector_modules = np.sqrt(self.v_r_vector[0] ** 2 + self.v_r_vector[1] ** 2 + self.v_r_vector[2] ** 2)
        self.v_b_vector_modules = np.sqrt(self.v_b_vector[0] ** 2 + self.v_b_vector[1] ** 2 + self.v_b_vector[2] ** 2)

        D_RB = [xb_ - x_, yb_ - y_, zb_ - z_] # Relative position vector
        D_BR = [x_ - xb_, y_ - yb_, z_ - zb_]
        D_RB_modules = np.sqrt(D_RB[0] ** 2 + D_RB[1] ** 2 + D_RB[2] ** 2)
        D_BR_modules = np.sqrt(D_BR[0] ** 2 + D_BR[1] ** 2 + D_BR[2] ** 2)
        
        # Angle between Relative position vector & velocity vector
        qr = np.arccos((D_RB[0] * self.v_r_vector[0] + D_RB[1] * self.v_r_vector[1] + D_RB[2] * self.v_r_vector[2]) / (
                    D_RB_modules * self.v_r_vector_modules)) 
        qb = np.arccos((D_BR[0] * self.v_b_vector[0] + D_BR[1] * self.v_b_vector[1] + D_BR[2] * self.v_b_vector[2]) / (
                    D_BR_modules * self.v_b_vector_modules))
        delh = z - zb

        # Safety zone check
        if v < self.v_min or v>self.v_max or z < self.h_min or z > self.h_max :
            self.done = True
            self.status = "Out of range"
            # print(v, z)
            r4 = -8
        else:
            # self.done = False
            r4 = 0
            
        # print("qr",qr)

       
        # if d0 <= self.d_max and np.abs(qr) < self.attk_max and np.abs(qb) < self.esp_max:
        #     self.done = True
        #     r1 = 8 #+ ((self.max_step-self.internal_step)/self.max_step)* 3
        #     self.status = "Red won"
        #     # print('Red win', "qr:", np.degrees(np.abs(qr)), np.degrees(self.attk_max), "qb>", np.degrees(np.abs(qb)), np.degrees(self.esp_max))
        # else:
        #     r1 = 0

        # # unfAVOURABLE SITUATION qB<
        # if d0 <= self.d_max and np.abs(qb) < self.attk_max:
        #     self.done = True
        #     r2 = -8
        #     self.status = "Blue won"
        #     # print('Blue win', "qb<", np.degrees(np.abs(qb)), np.degrees(self.attk_max))
        # else:
        #     r2 = 0
            
        if self.internal_step >= self.max_step:
            r3 = -5
            self.done = True
            # print("Done max  step")
            self.status = "max steps"
        else:
            r3 = -0.01


        # if qb > qr and np.abs(qb) < self.attk_max:
        #     if d0 > self.d_max:
        #         ra = (qb - qr) * np.exp(-(d0 - self.d_max) * (d0 - self.d_max) / (self.d_max * self.d_max))
        #     else:
        #         ra = (qb - qr) * 1
        # else:
        #     ra = 0

        if d0 <= self.d_max:
            self.done = True
            r1 = 8 #+ ((self.max_step-self.internal_step)/self.max_step)* 3
            self.status = "Red won"
            # print('Red win', "qr:", np.degrees(np.abs(qr)), np.degrees(self.attk_max), "qb>", np.degrees(np.abs(qb)), np.degrees(self.esp_max))
        else:
            r1 = 0
        reward = r1 #+ r2 + r3 + r4+(0.5*ra)

        v_x = self.v_r_vector[0] - self.v_b_vector[0]
        v_y = self.v_r_vector[1] - self.v_b_vector[1]
        v_z = self.v_r_vector[2] - self.v_b_vector[2]
        
        psi_D = np.arctan2(yb_ - y_, xb_ - x_) + (2 * np.pi if np.arctan2(yb_ - y_, xb_ - x_) < 0 else 0)
        theta_D = np.arcsin((zb_ - z_)/d0)


  

        observation_ = [(x_ - xb_)/self.D_th,
                        (y_ - yb_)/self.D_th,
                        z_/self.H_th,
                        d0/self.D_th,
                        theta_D/np.pi,
                        psi_D/ 2*np.pi,
                        v_x/(self.v_max-self.v_min),
                        v_y/(self.v_max-self.v_min),
                        v_z/(self.v_max-self.v_min), 
                        qr/np.pi,
                        qb/np.pi]


        # if self.internal_step >= self.max_step:
            

        return observation_, reward, self.done, self.status
    



    def get_reward(self, reward):
        self.r.append(reward)
        return self.r

    def get_state(self):
        x = self.state_red[0]
        y = self.state_red[1]
        z = self.state_red[2]
        v     = self.state_red[3]
        theta = self.state_red[4]
        psi = self.state_red[5]
        
        xb = self.state_blue[0]
        yb = self.state_blue[1]
        zb = self.state_blue[2]
        vb     = self.state_blue[3]
        thetab = self.state_blue[4]
        psib = self.state_blue[5]
        
        st = [x, y, z,v ,theta ,psi ,xb ,yb ,zb ,vb ,thetab ,psib] + [self.distance, self.height, self.disengagement_angle, self.deviation_angle, self.v_b_vector, self.v_r_vector]
        return st

    def reset(self):


        left =  [[np.random.uniform(200, 1800), np.random.uniform(1500, 3500), 4000, 60, 0, np.random.uniform(-0.5*np.pi, 0.5*np.pi)] ,[np.random.uniform(1500, 2200), np.random.uniform(1500, 3500), 4000, 60, 0, np.random.uniform(-0.3*np.pi, 0.3*np.pi)] ]

        right  = [[np.random.uniform(1900, 3500), np.random.uniform(1500, 3500), 4000, 60, 0, np.pi-np.random.uniform(-0.5*np.pi, 0.5*np.pi)] , [np.random.uniform(1500, 2200), np.random.uniform(1500, 3500), 4000, 60, 0, np.pi-np.random.uniform(-0.3*np.pi, 0.3*np.pi)] ]



        bottom =  [[np.random.uniform(1500, 3500), np.random.uniform(200, 1800), 4000, 60, 0, np.random.uniform(0, 1*np.pi)] ,
        [np.random.uniform(1500, 3500), np.random.uniform(1500, 2200), 4000, 60, 0, np.random.uniform(0.3*np.pi, 0.7*np.pi)] ]

        top =  [[np.random.uniform(1500, 3500), np.random.uniform(1900, 3500), 4000, 60, 0, np.pi-np.random.uniform(0, 1*np.pi)] ,

        [np.random.uniform(1500, 3500), np.random.uniform(1500, 2200), 4000, 60, 0, np.pi-np.random.uniform(0.3*np.pi, 0.7*np.pi)] ]

        pos = random.choice([top, bottom,left, right])
        self.state_red = pos[0]
        self.state_blue = pos[1]


        # Dis
        # self.state_red = [np.random.uniform(1500, 2200), np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-0.5*np.pi, 0.5*np.pi)] 
        # self.state_blue = [np.random.uniform(200, 1800), np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-0.3*np.pi, 0.3*np.pi)] 

        # Adv
        # self.state_red = [np.random.uniform(200, 1800), np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-0.5*np.pi, 0.5*np.pi)] 
        # self.state_blue = [np.random.uniform(1500, 2200), np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-0.3*np.pi, 0.3*np.pi)] 


        # adv= [np.random.uniform(200, 1800), np.random.uniform(1500, 2200)] 
        # dis = [np.random.uniform(1500, 2200), np.random.uniform(200, 1800)]

        # x_pos = random.choice([adv, dis])
        # self.state_red = [x_pos[0], np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-np.pi, np.pi)] 
        # self.state_blue = [x_pos[1], np.random.uniform(1500, 3500), np.random.uniform(700, 2500), 60, 0, np.random.uniform(-np.pi, np.pi)] 

        # self.state_red = [200, 2500, 700, 60, 0, 0] 
        # self.state_blue = [3500, 2500, 2500, 60, 0, 0] 

        x_, y_, z_, v_, theta_, psi_ = self.state_red
        xb_, yb_, zb_, vb_, thetab_, psib_ = self.state_blue


        d0 = np.sqrt((xb_ - x_) ** 2 + (yb_ - y_) ** 2 + (zb_ - z_) ** 2)
        self.v_r_vector = [v_ * np.cos(theta_) * np.cos(psi_), v_ * np.cos(theta_) * np.sin(psi_), v_ * np.sin(theta_)]
        self.v_b_vector = [vb_ * np.cos(thetab_) * np.cos(psib_), vb_ * np.cos(thetab_) * np.sin(psib_), vb_ * np.sin(thetab_)]
        self.v_r_vector_modules = np.sqrt(self.v_r_vector[0] ** 2 + self.v_r_vector[1] ** 2 + self.v_r_vector[2] ** 2)
        self.v_b_vector_modules = np.sqrt(self.v_b_vector[0] ** 2 + self.v_b_vector[1] ** 2 + self.v_b_vector[2] ** 2)

        D_RB = [xb_ - x_, yb_ - y_, zb_ - z_] # Relative position vector
        D_BR = [x_ - xb_, y_ - yb_, z_ - zb_]
        D_RB_modules = np.sqrt(D_RB[0] ** 2 + D_RB[1] ** 2 + D_RB[2] ** 2)
        D_BR_modules = np.sqrt(D_BR[0] ** 2 + D_BR[1] ** 2 + D_BR[2] ** 2)
        
        # Angle between Relative position vector & velocity vector
        qr = np.arccos((D_RB[0] * self.v_r_vector[0] + D_RB[1] * self.v_r_vector[1] + D_RB[2] * self.v_r_vector[2]) / (
                    D_RB_modules * self.v_r_vector_modules)) 
        qb = np.arccos((D_BR[0] * self.v_b_vector[0] + D_BR[1] * self.v_b_vector[1] + D_BR[2] * self.v_b_vector[2]) / (
                    D_BR_modules * self.v_b_vector_modules))
        delh = z_ - zb_

        v_x = self.v_r_vector[0] - self.v_b_vector[0]
        v_y = self.v_r_vector[1] - self.v_b_vector[1]
        v_z = self.v_r_vector[2] - self.v_b_vector[2]
        
        psi_D = np.arctan2(yb_ - y_, xb_ - x_) + (2 * np.pi if np.arctan2(yb_ - y_, xb_ - x_) < 0 else 0)
        theta_D = np.arcsin((zb_ - z_)/d0)

 
        observation = [(x_ - xb_)/self.D_th,
                        (y_ - yb_)/self.D_th,
                        z_/self.H_th,
                        d0/self.D_th,
                        theta_D/np.pi,
                        psi_D/ 2*np.pi,
                        v_x/(self.v_max-self.v_min),
                        v_y/(self.v_max-self.v_min),
                        v_z/(self.v_max-self.v_min), 
                        qr/np.pi,
                        qb/np.pi]
        
        self.done = False
        self.internal_step = 0
        
        return observation

    def draw(self, x, y, z, xb, yb, zb, v_b_vector_data, v_r_vector_data, episode, step, reward, visualize=False, save=False, save_dir="./results"):
        """
        Draws a 3D scene with two UAVs, vectors, and dynamically updating rotated arcs based on self.pusaib.
        
        Parameters:
            x, y, z: Coordinates of the red UAV over time.
            xb, yb, zb: Coordinates of the blue UAV over time.
            v_b_vector_data, v_r_vector_data: Velocity vector data for each UAV.
            episode, step, reward: Current episode, step, and reward values.
            visualize (bool): If True, shows real-time plot.
            save (bool): If True, saves the trajectory to a file.
            save_dir (str): Directory for saving files.
        """
        # Ensure arrays are of numpy type for consistency
        x, y, z = map(np.array, (x, y, z))
        xb, yb, zb = map(np.array, (xb, yb, zb))
        trajectory = np.array([x, y, z, xb, yb, zb, v_b_vector_data, v_r_vector_data], dtype=object)

        # Set scaling for vectors and arrowheads
        vector_scaling, head_scaling = 2, 5

        # Initialize the figure and 3D axis if not already created
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')
            # self.ax.set_xlim(0, 10000)
            # self.ax.set_ylim(0, 10000)
            self.ax.set_zlim(0, 10000)

            # Initialize connection line between UAVs
            self.connection_line, = self.ax.plot([], [], [], color="grey", linestyle="--", label="Connection")

            # Initialize quivers (vectors) for velocity
            self.vector_quiver_blue = self.ax.quiver(0, 0, 0, 0, 0, 0, color="blue", arrow_length_ratio=head_scaling)
            self.vector_quiver_red = self.ax.quiver(0, 0, 0, 0, 0, 0, color="red", arrow_length_ratio=head_scaling)

            # Placeholder for the arcs so they can be removed dynamically
            self.arc_collection_blue = None  
            self.arc_collection_red = None  

        # Clear the axis and reset the limits
        self.ax.clear()
        # self.ax.set_xlim(0, 10000)
        # self.ax.set_ylim(0, 10000)
        self.ax.set_zlim(0, 6000)

        # Re-plot the lines and points
        self.ax.scatter3D(x[0], y[0], z[0], color="black", s=20)
        self.ax.scatter3D(xb[0], yb[0], zb[0], color="green", s=20)

        # Update UAV lines
        self.red_line, = self.ax.plot(x, y, z, color="red", label="Red UAV")
        self.blue_line, = self.ax.plot(xb, yb, zb, color="blue", label="Blue UAV")

        # Update connection line between UAVs
        if x.size > 0 and xb.size > 0:
            self.connection_line.set_data([x[-1], xb[-1]], [y[-1], yb[-1]])
            self.connection_line.set_3d_properties([z[-1], zb[-1]])

        # Update quivers for velocity vectors
        if v_b_vector_data:
            v_b_vector = v_b_vector_data[-1]
            self.vector_quiver_blue.remove()  # Remove the old vector
            self.vector_quiver_blue = self.ax.quiver(
                xb[-1], yb[-1], zb[-1], 
                v_b_vector[0], v_b_vector[1], v_b_vector[2], 
                color="blue", length=vector_scaling * np.linalg.norm(v_b_vector), normalize=True, arrow_length_ratio=head_scaling
            )

        if v_r_vector_data:
            v_r_vector = v_r_vector_data[-1]
            self.vector_quiver_red.remove()  # Remove the old vector
            self.vector_quiver_red = self.ax.quiver(
                x[-1], y[-1], z[-1], 
                v_r_vector[0], v_r_vector[1], v_r_vector[2], 
                color="red", length=vector_scaling * np.linalg.norm(v_r_vector), normalize=True, arrow_length_ratio=head_scaling
            )

        # Update plot titles and labels
        self.ax.set_title(f"Episode: {episode}, Step: {step}, Reward: {reward}")
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')

        plt.draw()
        plt.pause(0.01)
        plt.show(block=visualize)

        # Save the trajectory data if save is True
        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(f"{save_dir}/Trajectory_{episode}_{step}.npy", trajectory)
           
            
    def render(self,state, mode=True, filepath='./JSBSimRecording.txt.acmi'):
        Xr, Yr, Zr, Vr, THETAr, PSUAIr, Xb, Yb, Zb, Vb, THETAb, PSUAIb = np.array(state)
        
        # print("Xr", Xr)
        # print("Xb", Xb)
        # print("V", Vr)
        # print("THETA", THETAr)
        # print("PSUAI", PSUAIr)
        Vx = Vr*np.cos(THETAr)*np.cos(PSUAIr) # vx = vcosθcosψ
        Vy = Vr*np.cos(THETAr)*np.sin(PSUAIr) # vy = vcosθsinψ
        
        lat_ref, long_ref = 120.0, 60.00
        uids = ['A0100', 'B0100']
        colors = ['Red', 'Blue']
        
        m_lat = 111132.92 - (559.82*np.cos(2*lat_ref)) + (1.175*np.cos(4*lat_ref)) - ((0.0023*np.cos(6*lat_ref)))
        m_long = (111412.84*np.cos(lat_ref)) - (93.5*np.cos(3*lat_ref)) + (0.118*np.cos(5*lat_ref))
        
        LONGr = (Xr / m_long) + long_ref
        LATr = (Yr / m_lat) + lat_ref

        LONGb = (Xb / m_long) + long_ref
        LATb = (Yb / m_lat) + lat_ref

        LONG, LAT, Z, THETA, PSUAI = [LONGr, LONGb], [LATr, LATb], [Zr, Zb], [THETAr, THETAb], [PSUAIr, PSUAIb]
        
        if mode == True:
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:

                for current_step in range(len(Xr)):
                
                    for i,uid in enumerate(uids):
                    
                        timestamp = current_step * 1 * 1
                        f.write(f"#{timestamp:.2f}\n")
                        # T = Longitude | Latitude | Altitude | Roll | Pitch | Yaw | U | V | Heading
                        lon = LONG[i][current_step]
                        lat = LAT[i][current_step]
                        alt = Z[i][current_step] #*0.3048
                        # roll = self.sim.get_property_value(STATE_FORMAT[9]) * 180 / np.pi
                        pitch = THETA[i][current_step] * 180 / np.pi
                        yaw = PSUAI[i][current_step] * 180 / np.pi
                        # roll = ROLL[current_step] * 180 / np.pi
                        # velocityX = Vx[current_step]
                        # velocityY = Vy[current_step]
                        log_msg = f"{uid},T={lat}|{lon}|{alt}||{pitch}|{yaw},"
                        
                        log_msg += f"Name=f16,"
                        log_msg += f"Color={colors[i]}"
                        if log_msg is not None:
                            f.write(log_msg + "\n")
                    
                print("DOne")
        else:
            raise NotImplementedError