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

#from nn_model import Actor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_size = 3
action_size =1
random_seed = 22
num_agents = 3
i_episode = 1

Ward1_model = Actor(state_size, action_size, random_seed).to(device)
Ward1_model.load_state_dict(torch.load('C:/Users/catch/Downloads/MTP 2021/Model/Ward1_actor.pth'))

Ward2_model = Actor(state_size, action_size, random_seed).to(device)
Ward2_model.load_state_dict(torch.load('C:/Users/catch/Downloads/MTP 2021/Model/Ward2_actor.pth'))

Ward3_model = Actor(state_size, action_size, random_seed).to(device)
Ward3_model.load_state_dict(torch.load('C:/Users/catch/Downloads/MTP 2021/Model/Ward3_actor.pth'))

def percentageCalculate(val):
    if val >=0 and val < 35:
        return 1
    elif val >=35 and val < 60:
        return 2
    elif val >=60:
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

def findactions(s1, s2, trend1, trend2, trend3):
    
    states = []
    trends = [trend1, trend2, trend3]
    actions = []
    
    w1_state = [s1, s2, trend1]
    states.append(w1_state)
    w2_state = [s1, s2, trend2]
    states.append(w2_state)
    w3_state = [s1, s2, trend3]
    states.append(w3_state)
    
    states = np.array([states])

    for i in range(3):
    
        if trends[i] < 2 and s1 ==1 and s2 ==1:
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

h_actions = {
    0 : "Turn on backup electricity generators for medical equiments",
    1: "Keep the ambulances,vehicles ready for the incoming patients",
    2: "Increase the number of Beds for the incoming patients",
    3: "No actions required to be taken for Hospital CI"
 }

e_actions = {
    0 : "Power incoming is at critical levels due to surge in current demand",
    1: "Alert maintenance vehicles about alternative routes",
    2: "Take measures to increase power output for future demand",
    3: "No actions required to be taken for Power CI"
}

t_actions = {
    0 : "Turn on backup electricity generators due to potential powercuts",
    1: "Share Alternative routes with all ambulances of the ward",
    2: "Alert! Potential transport-blockages due to Covid restrictions",
    3: "No action required to be taken for Transport CI "
  }


def main():


    trend1 = int(input("Enter covid trend at Ward 1 (range: 1-3) : "))
    p1 = percentageCalculate (float(input("Enter Power CI severity in percentage: ")))
    t1 = percentageCalculate (float(input("Enter Transport CI severity in percentage: ")))
    print(" ")

    trend2 = int(input("Enter covid trend at Ward 2 (range: 1-3) : "))
    p2 = percentageCalculate (float(input("Enter Power CI severity in percentage : ")))
    t2 = percentageCalculate (float(input("Enter Transport CI severity in percentage : ")))
    print(" ")

    trend3 = int(input("Enter covid trend at Ward 3 (range: 1-3) : "))
    p3 = percentageCalculate (float(input("Enter Power CI severity in percentage : ")))
    t3 = percentageCalculate (float(input("Enter Transport CI severity in percentage : ")))
    print(" ")


    w1_all_actions1 = findactions(p1, t1, trend1, trend2, trend3 )
    w1_all_actions2 = findactions(trend1, t1, trend1, trend2, trend3 )
    w1_all_actions3 = findactions(p1, trend1, trend1, trend2, trend3 )

    w2_all_actions1 = findactions(p2, t2, trend1, trend2, trend3 )
    w2_all_actions2 = findactions(trend2, t2, trend1, trend2, trend3 )
    w2_all_actions3 = findactions(p2, trend2, trend1, trend2, trend3 )

    w3_all_actions1 = findactions(p3, t3, trend1, trend2, trend3 )
    w3_all_actions2 = findactions(trend3, t3, trend1, trend2, trend3 )
    w3_all_actions3 = findactions(p3, trend3, trend1, trend2, trend3 )

    print("w1")
    print("Action for Hosptial : ", h_actions[w1_all_actions1[0]])
    print("Action for Power Station : ", e_actions[w1_all_actions2[1]])
    print("Action for Transport Office : ", t_actions[w1_all_actions3[2]])
    print("w2")
    print("Action for Hosptial : ", h_actions[w2_all_actions1[0]])
    print("Action for Power Station : ", e_actions[w2_all_actions2[1]])
    print("Action for Transport Office : ", t_actions[w2_all_actions3[2]])
    print("w3")
    print("Action for Hosptial : ", h_actions[w3_all_actions1[0]])
    print("Action for Power Station : ", e_actions[w3_all_actions2[1]])
    print("Action for Transport Office : ", t_actions[w3_all_actions3[2]])

if __name__ == "__main__":
    main()