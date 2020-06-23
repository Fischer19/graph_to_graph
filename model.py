import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

from torch import optim

SOS_token = 0
EOS_token = 1

# ---------------------- GNN encoder ----------------------------

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, num_rel, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.weight = nn.Parameter(torch.Tensor(num_rel, in_feats))

    def forward(self, g, feature, input_layer = False):
        
        def gcn_msg(edges):
            # The argument is a batch of edges.
            # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
            index = edges.data["rel_type"]
            return {'msg' : self.weight[index] + edges.src['h']}

        def gcn_reduce(nodes):
            # The argument is a batch of nodes.
            # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
            return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}
        
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
    
class Net(nn.Module):
    def __init__(self, input_size = 300, hidden_size = 256, output_size = 8):
        super(Net, self).__init__()
        self.gcn1 = GCN(input_size, 34, hidden_size, F.relu)
        self.gcn2 = GCN(hidden_size, 34, output_size, None)

    def forward(self, g, feature):
        x = self.gcn1(g, feature, True)
        x = self.gcn2(g, x)
        g_emb = torch.mean(x, axis = 0)
        return x, g_emb
    
    
    
# ---------------------- Hierarchical LSTM ----------------------------

class LSTM_node_generator(nn.Module):
    # hidden_size the same as GNN output
    def __init__(self, hidden_size, output_size):
        super(LSTM_node_generator, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)    
    
    
class LSTM_phrase_generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(LSTM_phrase_generator, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class LSTM_edge_generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(LSTM_edge_generator, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)    

# ------------------- graph-to-graph --------------------------

class graph_to_graph(nn.Module):
    # take a Commonsense graph as input, output generated new nodes(phrase) and edges(phrase)
    def __init__(self, input_size, hidden_size, node_output_size, phrase_output_size, edge_output_size):
        super(graph_to_graph, self).__init__()
        self.node = LSTM_node_generator(hidden_size, node_output_size)
        self.phrase = LSTM_phrase_generator(hidden_size, phrase_output_size) #TODO: replace by gpt-2
        self.edge = LSTM_edge_generator(hidden_size, edge_output_size)
        self.graph_encoder = Net(input_size, 256, hidden_size)
        
    def forward(self, g, feature, max_length = 100):
        # get graph embedding
        g_embedding, node_embedding = self.graph_encoder(g, feature)
        node_embedding = list(node_embedding)
        
        node_decoder_input = torch.tensor([[SOS_token]], device=device)
        node_decoder_hidden = g_embedding
        
        new_node_list = []
        new_phrase_list = []
        new_edge_list = []
        
        # TODO: implementing teacher-forcing
        for ni in range(max_length):
            node_decoder_output, node_decoder_hidden = self.node(
                node_decoder_input, node_decoder_hidden)
            new_node_embedding = node_decoder_hidden
            topv, topi = node_decoder_output.topk(1)
            node_decoder_input = topi.squeeze().detach()  # detach from history as input
            if node_decoder_input.item() == EOS_token: # stop generating node
                break
            else:  # new node embedding generated
                # add new node embedding to the list
                new_node_list.append(new_node_embedding)
                # generate new phrase
                new_phrase = []
                phrase_decoder_input = torch.tensor([[SOS_token]], device=device)
                for pi in range(max_length):
                    phrase_decoder_output, phrase_decoder_hidden = self.phrase(
                        phrase_decoder_input, phrase_decoder_hidden)

                    topv, topi = phrase_decoder_output.topk(1)
                    phrase_decoder_input = topi.squeeze().detach()  # detach from history as input

                    if phrase_decoder_input.item() == EOS_token: # stop generating node
                        break
                    new_phrase.append(phrase_decoder_input)
                    
                new_phrase_list.append(new_phrase)
        # generate edge between nodes

        for i, node1 in enumerate(node_embedding + new_node_list):
            for j, node2 in enumerate(node_embedding + new_node_list):
                edge_decoder_hidden = torch.cat([node1, node2])
                edge_decoder_input = torch.tensor([[SOS_token]], device=device)
                new_edge = []
                for ei in range(max_length):
                    edge_decoder_output, edge_decoder_hidden = self.edge(
                    edge_decoder_input, edge_decoder_hidden)
                    
                    topv, topi = edge_decoder_output.topk(1)
                    edge_decoder_input = topi.squeeze().detach()  # detach from history as input

                    if edge_decoder_input.item() == EOS_token: # stop generating node
                        break
                    new_edge.append(edge_decoder_input)
                new_edge_list.append((i,j,new_edge))
                
        return new_phrase_list, new_edge_list
    
    
if __name__ == "__main__":
    
    CORPUS_SIZE = 10000
    graph_generator = graph_to_graph(300, 256, 2, CORPUS_SIZE, CORPUS_SIZE)
        