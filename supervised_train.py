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


def evaluate(X,y):
    #print(torch.from_numpy(np.array(X)).shape, torch.transpose(y.cpu(), 0, 1).shape)
    count = torch.sum(torch.from_numpy(np.array(X)).view(y.shape[1], y.shape[0]) ==  torch.transpose(y.cpu(), 0, 1)).item()
    print("Acc: {:.4f}".format(count / (y.shape[0] * y.shape[1])))


def supervised_train(graph_encoder, node_decoder, iteration, X, y, batch_size = 10, learning_rate  = 1e-3, teacher_forcing = False):
    device = torch.device("cuda")
    graph_encoder.to(device)
    node_decoder.to(device)
    #batch_size = len(X)
    y = torch.Tensor(y).long().to(device)
    
    batch_graph = []
    target = []
    for bi in range(len(X) // batch_size):
        g_list = []
        for i in range(batch_size):    
            g = sub_graph_G.subgraph(X[i])
            g.copy_from_parent()
            g.ndata["x"] = g.ndata["x"].float().to(device)
            edge_norm = torch.ones(g.edata['rel_type'].shape[0]).to(device)
            g.edata.update({'norm': edge_norm.view(-1,1).to(device)})
            g_list.append(g)
        batch_graph.append(dgl.batch(g_list).to(device))
        target.append(y[bi * batch_size:(bi+1) * batch_size])
    
    graph_encoder_optimizer = optim.Adam(graph_encoder.parameters(), lr=learning_rate)
    node_decoder_optimizer = optim.Adam(node_decoder.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(iteration):
        loss = 0
        output = []
        for j in range(len(batch_graph)):
            graph_encoder_optimizer.zero_grad()
            node_decoder_optimizer.zero_grad()
            node_embedding, g_embedding = graph_encoder(batch_graph[j])
            node_decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(batch_size, 1)
            #node_decoder_hidden = (g_embedding.view(1,batch_size,-1), torch.zeros_like(g_embedding).view(1,batch_size,-1))
            node_decoder_hidden = (g_embedding.view(1,batch_size,-1), g_embedding.view(1,batch_size,-1))

            new_node_list = []

            # TODO: implementing teacher-forcing
            for ni in range(y.shape[1]):
            #for ni in range(1):
                node_decoder_output, node_decoder_hidden = node_decoder(
                    node_decoder_input, node_decoder_hidden)
                new_node_embedding = node_decoder_hidden
                topv, topi = node_decoder_output.topk(1)
                if teacher_forcing:
                    node_decoder_input = target[j][:,ni].view(batch_size, 1)
                else:
                    node_decoder_input = topi.squeeze().detach()  # detach from history as input
                '''
                if node_decoder_input.item() == EOS_token: # stop generating node
                    break
                '''
                #print(node_decoder_output, y[ni])
                loss += (1 + 2 * (ni == 0)) * criterion(node_decoder_output.view(batch_size,-1), target[j][:, ni].view(batch_size))
                output.append(topi.squeeze().detach().cpu().numpy())

                loss.backward(retain_graph = True)

                node_decoder_optimizer.step()
                graph_encoder_optimizer.step()
        
        if i % 100 == 0:
            evaluate(output, y)
            print("Iteration {}/{}, loss:{:.2f}".format(i, iteration, loss.item()))

if __name__ == "__main__":
    with open("DGL_graph.pkl", "rb") as f:
        g = pickle.load(f)
    
    with open("data/conceptnet/embedding_values.pkl", "rb") as f:
        embedding_values = pickle.load(f)
    print("-------------- Finish loading -------------")
    g.ndata["x"] = embedding_values

    # subsample a strongly-connected subgraph

    G_nx = g.to_networkx()
    sub_G_nx = nx.strongly_connected_components(G_nx)
    SCC = []
    for item in sub_G_nx:
        if len(item) > 2:
            SCC.append(item)
    component = list(SCC[0])
    print("-------------- Training data subsampled -------------")
    # assign embedding to graph
    sub_graph = g.subgraph(component)
    sub_graph.copy_from_parent()
    sub_graph_G = sub_graph

    X,y = gen_training_set(sub_graph_G, 3, 100)
    print("-------------- Training data generated -------------")
    CORPUS_SIZE = 10000
    input_size = 300
    hidden_size = 128
    node_output_size = g.ndata['x'].shape[0] + 2
    phrase_output_size = CORPUS_SIZE
    edge_output_size = CORPUS_SIZE
    num_rels = 34
    batch_size = 100
    n_hidden_layers = 2
    n_bases = -1
    device = torch.device("cuda")

    node_generator = LSTM_node_generator(hidden_size, node_output_size, batch_size)
    graph_encoder = Model(input_size,
                hidden_size,
                hidden_size,
                num_rels,
                num_bases=n_bases,
                num_hidden_layers=n_hidden_layers).to(device)
    print("-------------- Training start -------------")
    supervised_train(graph_encoder, node_generator, 5000, X[:batch_size* 1], y[:batch_size*1], 100, 1e-4, True)