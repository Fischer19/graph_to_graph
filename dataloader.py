# load data
import pickle
import torch
import dgl
import networkx as nx
import numpy as np
import copy

def create_graph_sampler(sub_graph_G, init_node = 1, expand_scale = 3, layer = 3):
    sub_graph_G.readonly()
    neighbor_sampler = dgl.contrib.sampling.sampler.NeighborSampler(sub_graph_G, init_node, expand_scale, layer)

    sampler_list = []
    for item in neighbor_sampler:
        sampler_list.append(item)
    return sampler_list

def gen_subgraph_nodes(sampler_list, index = 0, num = 15):
    nodeflow = sampler_list[index]
    node_list = []
    for i in range(nodeflow.num_layers - 1, -1, -1):
        #print(list(nodeflow.layer_parent_nid(i).numpy()))
        node_list = node_list + list(nodeflow.layer_parent_nid(i).numpy())
    return list(set(node_list))
    selected_node_list = set(node_list[:num])
    '''
    if len(selected_node_list) == num:
        return list(selected_node_list)
    else:
        start = num
        end = 2*num - len(selected_node_list)
        while True:
            selected_node_list = list(selected_node_list)
            selected_node_list = set(selected_node_list + node_list[start:end])
            start = copy.deepcopy(end)
            end = end + num - len(selected_node_list)
            if len(selected_node_list) == num:
                return list(selected_node_list)
    '''
def generate_neighbor(subgraph_nodes, graph_G, num_of_neighbor = 3):
    potential_nodes = torch.Tensor([]).long()
    for i, node in enumerate(subgraph_nodes): 
        out_edges = graph_G.out_edges(node)[1].numpy()
        in_edges = graph_G.in_edges(node)[0].numpy()
        if len(out_edges) > num_of_neighbor:
            out_edges = np.random.choice(out_edges, num_of_neighbor, replace = False)
        if len(in_edges) > num_of_neighbor:
            in_edges = np.random.choice(in_edges, num_of_neighbor, replace = False)
        potential_nodes = torch.cat([potential_nodes, torch.from_numpy(out_edges)])
        potential_nodes = torch.cat([potential_nodes, torch.from_numpy(in_edges)])
    return set(list(potential_nodes.numpy()))

def generate_neighbor_withsource(subgraph_nodes, graph_G, num_of_neighbor = 3):
    potential_nodes = torch.Tensor([]).long()
    potential_nodes_source = torch.Tensor([]).long()
    for i, node in enumerate(subgraph_nodes): 
        out_edges = np.array(list(set(graph_G.out_edges(node)[1].numpy()) - set(subgraph_nodes)))
        in_edges = np.array(list(set(graph_G.in_edges(node)[0].numpy()) - set(subgraph_nodes)))
        source = torch.Tensor([node]).long()
        if len(out_edges) >= num_of_neighbor:
            out_edges = np.random.choice(out_edges, num_of_neighbor, replace = False)
            potential_nodes_source  = torch.cat([potential_nodes_source, source])
        if len(in_edges) >= num_of_neighbor:
            in_edges = np.random.choice(in_edges, num_of_neighbor, replace = False)
            potential_nodes_source  = torch.cat([potential_nodes_source, source])
        potential_nodes = torch.cat([potential_nodes, torch.from_numpy(out_edges).long()])
        potential_nodes = torch.cat([potential_nodes, torch.from_numpy(in_edges).long()])
    return list(potential_nodes.numpy()), list(potential_nodes_source.numpy())

def gen_training_set(seed_graph, new_nodes_num, num = 512, nodewise = False):
    sampler_list = create_graph_sampler(seed_graph)
    X = []
    y = []
    count = 0
    reset_count = 0
    i = -1
    while count < num:
        if count % 100 == 0:
            print("{:.4f}% finished".format(100 * count / num))
        i += 1
        node_list = gen_subgraph_nodes(sampler_list, i)
        #print(len(node_list), len(potential_node))
        if nodewise:
            potential_node, potential_node_source = generate_neighbor_withsource(node_list, seed_graph, 1)
            #potential_node = potential_node - set(node_list)
            X.append(list(node_list))
            y.append((potential_node, potential_node_source))
            count += 1
        else:
            potential_node = generate_neighbor(node_list, seed_graph, 3)
            potential_node = potential_node - set(node_list)
            if len(node_list) == 15 and len(potential_node) >= new_nodes_num:
                potential_node = np.random.choice(np.array(list(potential_node)), new_nodes_num, replace = False)
                X.append(list(node_list))
                y.append(list(potential_node))
                count += 1
        """
        if reset_count > 99:
            sampler_list = create_graph_sampler(seed_graph)
            reset_count = 0
            i = -1
        """
    return X,y

if __name__ == "__main__":

    with open("DGL_graph.pkl", "rb") as f:
        g = pickle.load(f)

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
    print(len(X), len(y))
    print(X[0])