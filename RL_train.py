import dgl
import dgl.function as fn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np
import pickle
import networkx as nx
from dataloader import *
from RGCN import *
from model import *

# Policy gradient
from torch.distributions import Categorical

gamma = 1

class graph_to_graph(nn.Module):
    # take a Commonsense graph as input, output generated new nodes(phrase) and edges(phrase)
    def __init__(self, input_size, hidden_size, node_output_size, phrase_output_size, edge_output_size, num_rels, n_hidden_layers, n_bases = -1):
        super(graph_to_graph, self).__init__()
        self.node = LSTM_node_generator(hidden_size, node_output_size)
        #self.phrase = LSTM_phrase_generator(hidden_size, phrase_output_size) #TODO: replace by gpt-2
        #self.edge = LSTM_edge_generator(hidden_size, edge_output_size)
        # USE vanilla GCN
        #self.graph_encoder = Net(input_size, 256, hidden_size)
        # USE R-GCN
        self.graph_encoder = Model(input_size,
              hidden_size,
              hidden_size,
              num_rels,
              num_bases=n_bases,
              num_hidden_layers=n_hidden_layers)
        
    def generate_graph_embedding(self, g):
        return self.graph_encoder(g)
    
    def node_policy(self, *args):
        return self.node(*args)
       
    def generate_node_baseline(self, g):
        node_embedding, g_embedding = self.graph_encoder(g)
        node_embedding = list(node_embedding)
        
        node_decoder_input = torch.tensor([[SOS_token]], device=device)
        node_decoder_hidden = (g_embedding.view(1,1,-1), torch.zeros_like(g_embedding).view(1,1,-1))
        
        new_node_list = []
        
        # TODO: implementing teacher-forcing
        for ni in range(max_length):
            node_decoder_output, node_decoder_hidden = self.node(
                node_decoder_input, node_decoder_hidden)
            new_node_embedding = node_decoder_hidden
            topv, topi = node_decoder_output.topk(1)
            node_decoder_input = topi.squeeze().detach()  # detach from history as input
            print(node_decoder_input)
            if node_decoder_input.item() == EOS_token: # stop generating node
                break
            else:  # new node embedding generated
                new_node_list.append(node_decoder_input.item())
        return new_node_list

class Policy(nn.Module):
    # Wrap up the LSTM decisions
    def __init__(self):
        super(Policy, self).__init__()
        #self.state_space = env.observation_space.shape[0]
        #self.action_space = env.action_space.n

        self.gamma = gamma
        self.graph_generator =  graph_to_graph(input_size, 
                                 hidden_size, 
                                 node_output_size, 
                                 phrase_output_size, 
                                 edge_output_size, 
                                 num_rels, 
                                 n_hidden_layers, 
                                 n_bases = -1)
        
        # Episode policy and reward history 
        self.policy_history = torch.Tensor([])
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.action_history = []
        
    def forward(self, *args):    
            return self.graph_generator.node_policy(*args)

def select_action(policy, *args):
    # state: (h_i, c_i), h_i ~ (1,1,hid_dim)
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    d_action,state = policy(*args)
    c = Categorical(torch.exp(d_action))
    action = c.sample()
    policy.policy_history = policy.policy_history.to(device)

    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        #print(policy.policy_history, c.log_prob(action))
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).view(-1).cpu()])
    else:
        policy.policy_history = (c.log_prob(action))
    return action, state


def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    #print(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, rewards).mul(-1), -1))
    # Update network weights
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item)
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.Tensor()
    policy.reward_episode= []
    policy.action_history = []
    return loss.item()

def main(episodes):
    device = torch.device("cuda")
    running_reward = 10
    policy.to(device)
    for episode in range(episodes):
        #state = env.reset() # Reset environment and record the starting state
        # initial state:
        #policy = Policy(graph_generator)
        node_embedding, g_embedding = policy.graph_generator.generate_graph_embedding(graph_S.to(device))
        node_decoder_input = torch.tensor([[SOS_token]], device=device)
        node_decoder_hidden = g_embedding
        node_decoder_hidden = (g_embedding.view(1,1,-1), torch.zeros_like(g_embedding).view(1,1,-1))

        for time in range(3):
            action, state = select_action(policy, node_decoder_input, node_decoder_hidden)
            policy.action_history.append(action.item())
            # Step through environment using chosen action
            #state, reward, done, _ = env.step(action.data[0])
            #reward = compute_reward(policy.action_history, y[0])
            node_decoder_input = action
            node_decoder_hidden = state
            # Save reward
            if action.item() == EOS_token: # stop generating node
                reward = compute_reward(policy.action_history, y[0])
                policy.reward_episode.append(reward)
                break
            reward = 0
            policy.reward_episode.append(reward)
        if time == 2:
            reward = compute_reward(policy.action_history, y[0])
            policy.reward_episode[-1] = reward
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)
        action_history = policy.action_history
        loss = update_policy()
        if episode % 50 == 0:
            print(action_history)
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}\t Loss: {:.4f}'.format(episode, time, running_reward, loss))

def compute_reward(action_history, y):
    #print(action_history, y)
    score = len(set(y).intersection(set(action_history)))
    return score

if __name__ == "__main__":

    with open("DGL_graph.pkl", "rb") as f:
        g = pickle.load(f)
    
    with open("data/conceptnet/embedding_values.pkl", "rb") as f:
        embedding_values = pickle.load(f)
            
    g.ndata["x"] = embedding_values

    # subsample a strongly-connected subgraph

    G_nx = g.to_networkx()
    sub_G_nx = nx.strongly_connected_components(G_nx)
    SCC = []
    for item in sub_G_nx:
        if len(item) > 2:
            SCC.append(item)
    component = list(SCC[0])

    # assign embedding to graph
    sub_graph = g.subgraph(component)
    sub_graph.copy_from_parent()
    sub_graph_G = sub_graph

    X,y = gen_training_set(sub_graph_G, 3, 100)

    graph_S = sub_graph_G.subgraph(X[0])
    graph_S.copy_from_parent()

    graph_S.ndata["x"] = graph_S.ndata["x"].float()
    edge_norm = torch.ones(graph_S.edata['rel_type'].shape[0])
    graph_S.edata.update({'norm': edge_norm.view(-1,1)})


    CORPUS_SIZE = 10000
    input_size = 300
    hidden_size = 16
    node_output_size = g.ndata['x'].shape[0] + 2
    phrase_output_size = CORPUS_SIZE
    edge_output_size = CORPUS_SIZE
    num_rels = 34
    n_hidden_layers = 2
    n_bases = -1
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=5e-3)

    main(1000)