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
from torch.utils.tensorboard import SummaryWriter

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




def plot_2d(fig, ax , vdata, thetadata, pusaidata, vbdata, thetabdata, pusaibdata, distancedata, heightdata, disengagement_angledata, deviation_angledata,scaling_const_data, episode, step,status,visualize =True, save=False, save_dir="./results"):

    params = np.array([vdata, thetadata, pusaidata, vbdata, thetabdata, pusaibdata, distancedata, heightdata, disengagement_angledata, deviation_angledata, scaling_const_data])
    
    
    if visualize:
        
        for row in ax:
            for a in row:
                a.clear()


        ax[0, 0].plot(vdata, 'r-', label='Velocity Agent')
        ax[0, 0].plot(vbdata, 'b-', label='Velocity Target')
        ax[0, 0].set(title='Speed', xlabel='Time Step', ylabel='Velocity (m/s)')
        ax[0, 0].legend(loc='upper left')
        ax[0, 0].grid(True)

        ax[0, 1].plot(distancedata, color='black', linestyle='--', label='Distance b/w target and agent')
        ax[0, 1].plot(heightdata, 'g--', label='Height difference')
        ax[0, 1].set(title='Distance', xlabel='Time Step', ylabel='Distance')
        ax[0, 1].legend(loc='upper left')
        ax[0, 1].grid(True)

        ax[1, 0].plot(np.degrees(pusaidata), 'r-', label='Yaw Agent')
        ax[1, 0].plot(np.degrees(pusaibdata), 'b-', label='Yaw Target')
        ax[1, 0].plot(np.degrees(thetadata), 'r--', label='Pitch Agent')
        ax[1, 0].plot(np.degrees(thetabdata), 'b--', label='Pitch Target')
        
        ax[1, 0].plot(scaling_const_data, color='black', linestyle='--', label='qb-qr exp term x100')
        
        ax[1, 0].set(title='Yaw and Pitch', xlabel='Time Step', ylabel='Degrees')
        ax[1, 0].legend(loc='upper left')
        ax[1, 0].grid(True)
        
        ax[1, 1].plot(np.degrees(disengagement_angledata), 'g--', label='q_r')
        ax[1, 1].plot(np.degrees(deviation_angledata), 'y--', label='q_b')
        

        ax[1, 1].plot(np.array(np.degrees(deviation_angledata))-np.array(np.degrees(disengagement_angledata)), color='black', linestyle='--', label='q_b-q_r')
        
        ax[1, 1].set(title='Angles', xlabel='Time Step', ylabel='Degrees')
        ax[1, 1].legend(loc='center left')
        ax[1, 1].grid(True)
        
        fig.suptitle(f"Episode: {episode}, Step: {step} Status: {status}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to make room for the suptitle
    plt.draw()
    plt.pause(0.1)
    # plt.show(block=True)
    
    # if save:
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     np.save(f"{save_dir}/Params_{episode}_{step}_{status}.npy", params)
        # Uncomment to save the figure
        # fig.savefig(f"{save_dir}/2d_plot_{episode}_{step}.jpg")


def train(args):
    start = time.time()
    env = Uav_Env()
    # Initialize DQN model
    if args.n is not None and os.path.exists(args.n):
        print(f"Loading model from {args.n}")
        dqn = torch.load(args.n)
    else:
        print("Model not loaded")
        return

        
    max_step = 100# env.max_step   #Maximum steps per episode
    reward_all = []
    matplotlib.pyplot.ion()
    writer = SummaryWriter()  #
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)
    
    
    
    tot_step = 0
    trajectory = []
    max_expisode = 1
    succ = 0 
    unscc = 0
    max_time = 0
    out_range =0
    for episode in range(1,max_expisode+1):
        observation= env.reset()  #Initial episode, reset the scene
        # Initialize lists for storing data
        s_store = []

        xdata, ydata, zdata = [], [], []
        vdata, thetadata, pusaidata, rolldata = [], [], [], []

        xbdata, ybdata, zbdata = [], [], []
        vbdata, thetabdata, pusaibdata = [], [], []
        v_b_vector_data, v_r_vector_data, scaling_const_data = [], [], []

        distancedata, heightdata, disengagement_angledata, deviation_angledata = [], [], [], []
        
        ep_reward = 0

        for step in range(max_step):
            
            while True:
                
                observation = np.array(observation)

                
                q_values = dqn.policy_net(torch.tensor([observation], dtype=torch.float32))
                action1 = q_values.max(1)[1].item()
                # action1, epsilon_val = dqn.choose_action(observation)  #choose_action function gives actions based on observations
                break


            observation_,  reward, done, status = env.step(action1) #The agent obtains information such as status and rewards at the next moment
            #state1 = env.get_state()
            # print(observation_)
            
           
            dqn.store_transition(observation, action1, reward, observation_,done) #Experience pool storage

            print(f"Step: {step} Action: {action1} Reward: {reward} Value: {q_values[0][action1].detach()}")


            observation = observation_
            # loss = dqn.learn()
            # writer.add_scalar('Loss/MSE', loss, tot_step)
            tot_step+=1
            ep_reward += reward
            
            # if (episode + 1) % TARGET_UPDATE == 0:  #Agent Target Network Update
            #     dqn.target_net.load_state_dict(dqn.policy_net.state_dict())
        

            if done == True:
                # print('episode：',episode,   'step:',step,   'reward：',ep_reward,'Out of safety range：',done, "epsilon_val", epsilon_val)
                print(f"'episode {episode}, succ {succ}, unscc {unscc}, max_time {max_time}, out_range {out_range} ")
                
                writer.add_scalar('Agent/Reward', ep_reward, episode)
                reward_all.append(ep_reward)
        
            
            x, y, z,v ,theta ,pusai ,xb ,yb ,zb ,vb ,thetab ,pusaib, distance, height, disengagement_angle, deviation_angle, v_b_vector, v_r_vector = env.get_state()
            scaling_const = 0
            
            xdata.append(x)
            ydata.append(y)
            zdata.append(z)
            vdata.append(v)
            thetadata.append(theta)
            pusaidata.append(pusai)
            # rolldata.append(roll)
            
            
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
            scaling_const_data.append(scaling_const)
            
            
            
            
            #if step == max_step - 1 or suc == True or done == True or uns == True:
            
            # env.draw(xdata, ydata, zdata,vdata ,thetadata ,pusaidata ,xbdata ,ybdata ,zbdata ,vbdata ,thetabdata ,pusaibdata, episode, step)
            # # if suc == True:
            #     env.draw(X, Y, Z, XB, YB, ZB)
            
            # if (((episode-1) %500 == 0) or (episode %500 == 0)or ((episode+1) %500 == 0)) and (episode>0):
            # env.draw(xdata, ydata, zdata, xbdata, ybdata, zbdata,v_b_vector_data, v_r_vector_data, episode, step, ep_reward,visualize =True, save= True)
                # plot_2d(fig, ax, vdata, thetadata, pusaidata, vbdata, thetabdata, pusaibdata,distancedata, heightdata, disengagement_angledata, deviation_angledata, scaling_const_data, episode, step,status, visualize =True, save= done == True or suc ==True or uns == True)
                       
            
            if done == True or step==(max_step-1):
                # dqn.save_checkpoint("file.pth")
                # env.draw(xdata, ydata, zdata, xbdata, ybdata, zbdata)
                # data = [xdata, ydata, zdata, xbdata, ybdata, zbdata, v_b_vector_data, v_r_vector_data, vdata, thetadata, pusaidata, vbdata, thetabdata, pusaibdata, distancedata, heightdata, disengagement_angledata, deviation_angledata, scaling_const_data]
                # trajectory.append(data)
                
                # save = True
                # print(episode, max_expisode)
                # if save and (episode==max_expisode):
                #     if not os.path.exists("./results"):
                #         os.makedirs("./results")
                #     np.save(f"./results/Trajectory_2.npy", np.array(trajectory, dtype=object))
                state = (xdata, ydata, zdata, vdata, thetadata, pusaidata, xbdata, ybdata, zbdata, vbdata, thetabdata, pusaibdata)
                env.render(state)

                # plot_2d(fig, ax, vdata, thetadata, pusaidata, vbdata, thetabdata, pusaibdata,distancedata, heightdata, disengagement_angledata, deviation_angledata, scaling_const_data, episode, step,status, visualize =True, save= done == True or suc ==True or uns == True)
                       
                env.draw(xdata, ydata, zdata, xbdata, ybdata, zbdata,v_b_vector_data, v_r_vector_data, episode, step, ep_reward,visualize =True, save= True)
                break


    print('DQN saved')
    end = time.time()
    print(end - start)


    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train or load DQN model")
    parser.add_argument('--n', type=str, default=None, help="Path to pre-trained model (optional)")
    args = parser.parse_args()
    train(args)