from DQN import DQN
from env_pre import Uav_Env
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import torch
import random
import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
# from statistics import blue_brain


TARGET_UPDATE = 4  #Update frequency of the target network

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


 
def train(args):
    start = time.time()
    env = Uav_Env()
    if args.n is not None and os.path.exists(args.n):
        print(f"Loading model from {args.n}")
        dqn = torch.load(args.n)
        # statistic = blue_brain
    else:
        print("Model not loaded")
        dqn = DQN()
  

    max_step = env.max_step   #Maximum steps per episode
    reward_all = []
    matplotlib.pyplot.ion()
    writer = SummaryWriter()  #
    # fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)
    
    # torch.save(dqn,f'attack_{0}_{0}.pkl') 
    
    tot_step = 0
    num_succes, num_fail, num_draw, num_range_out = 0, 0,0,0

    for episode in range(1,30001):
        observation = env.reset()  #Initial episode, reset the scene
        #

        xdata, ydata, zdata = [], [], []
        vdata, thetadata, pusaidata = [], [], []

        xbdata, ybdata, zbdata = [], [], []
        vbdata, thetabdata, pusaibdata = [], [], []
        v_b_vector_data, v_r_vector_data = [], []

        distancedata, heightdata, disengagement_angledata, deviation_angledata = [], [], [], []
        
        ep_reward = 0
        for step in range(max_step+1):
        
            action1, epsilon_val = dqn.choose_action(np.array(observation))  #choose_action function gives actions based on observations
            
            
    
            nxt_observation, reward, done, status = env.step(action1) #The agent obtains information such as status and rewards at the next moment
            
            q_values = dqn.policy_net(torch.FloatTensor(observation))
            
            
            dqn.store_transition(observation, action1, reward, nxt_observation, done) #Experience pool storage
            observation = nxt_observation
            
            loss = dqn.learn()
            writer.add_scalar('Loss/MSE', loss, tot_step)
            tot_step+=1
            ep_reward += reward
            
            if (episode + 1) % TARGET_UPDATE == 0:  #Agent Target Network Update
                dqn.target_net.load_state_dict(dqn.policy_net.state_dict())




        
            
            x, y, z,v ,theta ,pusai ,xb ,yb ,zb ,vb ,thetab ,pusaib, distance, height, disengagement_angle, deviation_angle, v_b_vector, v_r_vector = env.get_state()
     
            xdata.append(x)
            ydata.append(y)
            zdata.append(z)
            vdata.append(v)
            thetadata.append(theta)
            pusaidata.append(pusai)
     
            xbdata.append(xb)
            ybdata.append(yb)
            zbdata.append(zb)
            vbdata.append(vb)
            thetabdata.append(thetab)
            pusaibdata.append(pusaib)
            
            distancedata.append(distance)
            heightdata.append(height)
            disengagement_angledata.append(disengagement_angle)
            deviation_angledata.append(deviation_angle)
            v_b_vector_data.append(v_b_vector)
            v_r_vector_data.append(v_r_vector)

            # env.draw(xdata, ydata, zdata, xbdata, ybdata, zbdata,v_b_vector_data, v_r_vector_data, episode, step, ep_reward,visualize =False, save= True)    
            
            
            if (done == True) and (episode%1000==0):
                env.draw(xdata, ydata, zdata, xbdata, ybdata, zbdata,v_b_vector_data, v_r_vector_data, episode, step, ep_reward,visualize =False, save= True)             
                torch.save(dqn,f'weights/attack_{episode}.pkl') 
            if (done == True):
                if status== "Out of range":
                    num_range_out+=1
                if status== "Red won":
                    num_succes+=1
                if status== "Blue won":
                    num_fail+=1
                if status== "max steps":
                    num_draw+=1 
                    
                print('\nepisode：',episode,   'step:',step,   'reward：',ep_reward, "Loss",loss, "epsilon_val", epsilon_val)
                print('Red won:',num_succes,   'Blue won:',num_fail,   'max steps:',num_draw, "Out of range:", num_range_out)
                writer.add_scalar('Agent/ep_reward', ep_reward, episode)
                writer.add_scalar('Agent/step', step, episode)
                writer.add_scalar('Rate/redwin', num_succes, episode)
                writer.add_scalar('Rate/bluwwin', num_fail, episode)
                writer.add_scalar('Rate/Draw', num_draw, episode)
                writer.add_scalar('Rate/range_out', num_range_out, episode)
                reward_all.append(ep_reward)               

                break

    
    print('DQN saved')
    end = time.time()
    print(end - start)

    plt.plot(np.arange(len(reward_all)), reward_all)
    np.save("rewards.npy", reward_all)
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train or load DQN model")
    parser.add_argument('--n', type=str, default=None, help="Path to pre-trained model (optional)")
    args = parser.parse_args()
    train(args)