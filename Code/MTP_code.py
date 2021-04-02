# -*- coding: utf-8 -*-
"""MTP work 25-03-2021.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uZwxR4sBjbiu9bi6hqvamsKVoH34GNV7
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_values(layer):
    wts = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(wts)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed=0, fc1_dims=256, fc2_dims=128):

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_values(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_values(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):

        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):

    def __init__(self, full_state_size, actions_size, seed=0, fcs1_units=256, fc2_dims=128):
       
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+actions_size, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.bn2 = nn.BatchNorm1d(fc2_dims)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_values(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_values(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):

        xs = F.relu(self.fcs1(state))
        xs = self.bn1(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

import numpy as np
import torch

#from nn_model import Actor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_size = 3
action_size =1
random_seed = 22
num_agents = 3
i_episode = 1

Ward1_model = Actor(state_size, action_size, random_seed).to(device)
Ward1_model.load_state_dict(torch.load('/content/sample_data/Ward1_actor.pth'))

Ward2_model = Actor(state_size, action_size, random_seed).to(device)
Ward2_model.load_state_dict(torch.load('/content/sample_data/Ward2_actor.pth'))

Ward3_model = Actor(state_size, action_size, random_seed).to(device)
Ward3_model.load_state_dict(torch.load('/content/sample_data/Ward3_actor.pth'))

def percentageCalculate(depth):
    if depth >=0 and depth < 35:
        return 1
    elif depth >=35 and depth < 60:
        return 2
    elif depth >=60:
        return 3

def act(state, i):
        if (i==0):
            state = torch.from_numpy(state).float().to(device)
            Ward2_model.eval()
            action = Ward2_model(state).cpu().data.numpy()
        elif (i==1):
            state = torch.from_numpy(state).float().to(device)
            Ward3_model.eval()
            action = Ward3_model(state).cpu().data.numpy()
        elif (i==2):
            state = torch.from_numpy(state).float().to(device)
            Ward1_model.eval()
            action = Ward1_model(state).cpu().data.numpy()
        return action

def findactions(beds, stall, trend1, trend2, trend3):
    
    states = []
    depths = [trend1, trend2, trend3]
    actions = []
    
    w1_state = [beds, stall, trend1]
    states.append(w1_state)
    w2_state = [beds, stall, trend2]
    states.append(w2_state)
    w3_state = [beds, stall, trend3]
    states.append(w3_state)
    
    states = np.array([states])

    for i in range(3):
    
        if depths[i] < 2 and beds ==1 and stall ==1:
            ac = 3
        else:
            action = act(states[0][i], i)
            action = action[0][0]
            if( -1.0 <= action < -0.3):
                ac = 0
            elif(-0.3 <= action < 0.3):
                ac = 1
            elif(0.3 <= action <=1.0):
                ac = 2
                
        actions.append(ac)
        
    return actions

w1_actions = {
  0: "Increase Capacity of Beds",
  1: "Increase stall and testing kits",
  2: "Keep the ambulances ready for incoming patient",
  3: "No action needed"
}

w2_actions = {
  0: "Increase Capacity of Beds",
  1: "Increase stall and testing kits",
  2: "Keep the ambulances ready for incoming patient",
  3: "No action needed"
}

w3_actions = {
  0: "Increase Capacity of Beds",
  1: "Increase stall and testing kits",
  2: "Keep the ambulances ready for incoming patient",
  3: "No action needed"
}

def main():

    all_actions = []

    trend1 = int(input("Enter covid trend at Ward 1: "))
    beds1 = percentageCalculate (float(input("Enter Number of beds available in percentage: ")))
    stall1 = percentageCalculate (float(input("Enter stall available in percentage: ")))
    print(" ")

    trend2 = int(input("Enter covid trend at Ward 2: "))
    beds2 = percentageCalculate (float(input("Enter Number of beds available in percentage: ")))
    stall2 = percentageCalculate (float(input("Enter stall available in percentage: ")))
    print(" ")

    trend3 = int(input("Enter covid trend at Ward 3: "))
    beds3 = percentageCalculate (float(input("Enter Number of beds available in percentage: ")))
    stall3 = percentageCalculate (float(input("Enter stall available in percentage: ")))
    print(" ")

    all_actions = findactions(beds1, stall1, trend1, trend2, trend3 )
    all_actions2 = findactions(beds2, stall2, trend1, trend2, trend3 )
    all_actions3 = findactions(beds3, stall3, trend1, trend2, trend3 )

    if all_actions[0] >= all_actions2[1] and all_actions[0] >= all_actions3[2]:
      print("Move the incoming patients to Hostipal in Ward 1")
    if all_actions2[1] >= all_actions[0] and all_actions2[1] >= all_actions3[2]:
      print("Move the incoming patients to Hostipal in Ward 2")
    if all_actions3[2] >= all_actions2[1] and all_actions3[2] >= all_actions[0]:
      print("Move the incoming patients to Hostipal in Ward 3")

    print(" ")
    print("Action for Ward 1 : ", w1_actions[all_actions[0]])
    print("Action for Ward 2 : ", w2_actions[all_actions2[1]])
    print("Action for Ward 3 : ", w3_actions[all_actions3[2]])
  
if __name__ == "__main__":
    main()