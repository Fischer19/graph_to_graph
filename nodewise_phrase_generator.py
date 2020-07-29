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
import argparse


def gen_phrase(phrase_decoder, new_node_embedding, target_id):
    loss = 0
    index = target_id
    y_tokenize = torch.Tensor(tokenizer(index2phrase[index])['input_ids'][1:]).view(batch_size,-1).long().to(device)
    output = []
    phrase_decoder_input = torch.tensor([CLS_token], device=device).view(1, 1)
    phrase_decoder_hidden = new_node_embedding
    for ni in range(y_tokenize.shape[1]):
        phrase_decoder_output, node_decoder_hidden = phrase_generator(
            phrase_decoder_input, phrase_decoder_hidden)
        topv, topi = phrase_decoder_output.topk(1)
        if teacher_forcing:
            phrase_decoder_input = y_tokenize[:,ni].view(batch_size, 1)
        else:
            phrase_decoder_input = topi.squeeze().detach()  # detach from history as input 

        loss += criterion(phrase_decoder_output.view(batch_size,-1), y_tokenize[:, ni].view(batch_size))
        output.append(topi.squeeze().detach().cpu().numpy())
    return loss, tokenizer.decode(output), y_tokenize

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--train', action = "store_true")
    parser.add_argument('-hd', '--hidden_size', required = False, type = int, default = 64)
    parser.add_argument('-e', '--embedding', required = False, default = "conceptnet")
    args = parser.parse_args()


    with open("DGL_graph.pkl", "rb") as f:
        g = pickle.load(f)


    if args.embedding == "BERT":
        embedding_values = torch.load("data/conceptnet/bert_embedding_pooled.pt")
    else:
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

    #X,y = gen_training_set(sub_graph_G, 3, 1000, nodewise=True)
    with open("nodewise_training_data.pkl", "rb") as f:
        dic = pickle.load(f)
        X = dic["X"]
        y = dic["y"]

    from transformers import BertTokenizer
    tokenizer = BertTokenizer("data/conceptnet/embedding_keys.txt")

    with open("data/conceptnet/embedding_keys.pkl", "rb") as f:
        embedding_keys = pickle.load(f)
    for i,item in enumerate(embedding_keys):
        words = item.replace("_", " ")
        embedding_keys[i] = words
    embedding_keys = np.array(embedding_keys)
    index2phrase = embedding_keys[component]

    CORPUS_SIZE = len(tokenizer.vocab)
    input_size = 300
    hidden_size = args.hidden_size
    node_output_size = 2
    phrase_output_size = CORPUS_SIZE
    num_rels = 34
    batch_size = 1
    num_batch = 2
    n_hidden_layers = 2
    n_bases = -1
    device = torch.device("cuda")

    node_generator = LSTM_node_generator(hidden_size * 2, node_output_size, batch_size)
    graph_encoder = Model(input_size,
                hidden_size,
                hidden_size,
                num_rels,
                num_bases=n_bases,
                num_hidden_layers=n_hidden_layers).to(device)
    phrase_generator = LSTM_phrase_generator(hidden_size * 2, phrase_output_size, 34, False)

    node_generator.load_state_dict(torch.load("parameters/nodewise_node_decoder_parameters_input300_h{}.pth".format(args.hidden_size)))
    graph_encoder.load_state_dict(torch.load("parameters/nodewise_graph_encoder_parameters_input300_h{}.pth".format(args.hidden_size)))
    phrase_generator.load_state_dict(torch.load("parameters/nodewise_phrase_generator_parameters_input300_h{}.pth".format(args.hidden_size)))
    graph_encoder.to(device)
    node_generator.to(device)
    phrase_generator.to(device)

    CLS_token = 0
    SEP_token = 1

    g_list = []
    y_list = []
    for i in range(200):
        g = sub_graph_G.subgraph(X[i])
        g.copy_from_parent()
        g.ndata["x"] = g.ndata["x"].float().to(device)
        edge_norm = torch.ones(g.edata['rel_type'].shape[0]).to(device)
        g.edata.update({'norm': edge_norm.view(-1,1).to(device)})
        g_list.append(g)
        y_list.append(y[i])

    if args.train:
        #CUDA_LAUNCH_BLOCKING=1
        learning_rate = 1e-3
        teacher_forcing = True
        target = torch.Tensor([[0,1]]).to(device).long()

        graph_encoder_optimizer = optim.SGD(graph_encoder.parameters(), lr=learning_rate)
        node_decoder_optimizer = optim.SGD(node_generator.parameters(), lr=learning_rate)
        phrase_decoder_optimizer = optim.SGD(phrase_generator.parameters(), lr=learning_rate)

        for i in range(200):
            for j in range(100):
                node_decoder_optimizer.zero_grad()
                graph_encoder_optimizer.zero_grad()
                phrase_decoder_optimizer.zero_grad()
                
                node_embedding, g_embedding = graph_encoder(g_list[j])
                output = []
                phrase_output = []
                criterion = nn.CrossEntropyLoss()
                    
                loss = 0
                for k in range(len(X[j])):
                    stack_embedding = torch.cat([node_embedding[k].view(1, -1), g_embedding])
                    node_decoder_input = torch.tensor([[CLS_token] * batch_size], device=device).view(batch_size, 1)
                    node_decoder_hidden = (stack_embedding.view(1,batch_size,-1), stack_embedding.view(1,batch_size,-1))
                    if X[j][k] not in y[j][1]:
                        target = torch.Tensor([[1]]).to(device).long()
                    else:
                        target = torch.Tensor([[0,1]]).to(device).long()
                    # Generate new nodes
                    for ni in range(target.shape[1]):
                        node_decoder_output, node_decoder_hidden = node_generator(
                            node_decoder_input, node_decoder_hidden)
                        new_node_embedding = node_decoder_hidden
                        topv, topi = node_decoder_output.topk(1)
                        if teacher_forcing:
                            node_decoder_input = target[:,ni].view(batch_size, 1)
                        else:
                            node_decoder_input = topi.squeeze().detach()  # detach from history as input 
                        output.append(topi.squeeze().detach().cpu().numpy())
                        loss += criterion(node_decoder_output.view(batch_size,-1), target[:, ni].view(batch_size))
                        # find the index of the corresponding new nodes
                        if node_decoder_input == 1:
                            break
                        n_id = y[j][1].index(X[j][k])
                        target_id = y[j][0][n_id]
                        phrase_loss, p_output, _ = gen_phrase(phrase_generator, new_node_embedding, target_id)
                        phrase_output.append(p_output)
                        loss += phrase_loss

                loss.backward(retain_graph = True)
                node_decoder_optimizer.step()
                graph_encoder_optimizer.step()
                phrase_decoder_optimizer.step()
            if i % 10 == 0:
                print(loss, phrase_output)

        torch.save(node_generator.state_dict(), "parameters/nodewise_node_decoder_parameters_input300_h{}.pth".format(args.hidden_size))
        torch.save(graph_encoder.state_dict(), "parameters/nodewise_graph_encoder_parameters_input300_h{}.pth".format(args.hidden_size))
        torch.save(phrase_generator.state_dict(), "parameters/nodewise_phrase_generator_parameters_input300_h{}.pth".format(args.hidden_size))
    else:
        teacher_forcing = True
        target = torch.Tensor([[0,1]]).to(device).long()

        count = 0
        total = 0
        for j in range(100):
            node_embedding, g_embedding = graph_encoder(g_list[j])
            output = []
            phrase_output = []
            target_output = []
            criterion = nn.CrossEntropyLoss()

            loss = 0
            for k in range(len(X[j])):
                stack_embedding = torch.cat([node_embedding[k].view(1, -1), g_embedding])
                node_decoder_input = torch.tensor([[CLS_token] * batch_size], device=device).view(batch_size, 1)
                node_decoder_hidden = (stack_embedding.view(1,batch_size,-1), stack_embedding.view(1,batch_size,-1))
                if X[j][k] not in y[j][1]:
                    target = torch.Tensor([[1]]).to(device).long()
                else:
                    target = torch.Tensor([[0,1]]).to(device).long()
                # Generate new nodes
                for ni in range(target.shape[1]):
                    node_decoder_output, node_decoder_hidden = node_generator(
                        node_decoder_input, node_decoder_hidden)
                    new_node_embedding = node_decoder_hidden
                    topv, topi = node_decoder_output.topk(1)
                    if teacher_forcing:
                        node_decoder_input = target[:,ni].view(batch_size, 1)
                    else:
                        node_decoder_input = topi.squeeze().detach()  # detach from history as input 
                    output.append(topi.squeeze().detach().cpu().numpy())
                    loss += criterion(node_decoder_output.view(batch_size,-1), target[:, ni].view(batch_size))
                    # find the index of the corresponding new nodes
                    if node_decoder_input == 1:
                        break
                    n_id = y[j][1].index(X[j][k])
                    target_id = y[j][0][n_id]
                    phrase_loss, p_output, y_tokenize = gen_phrase(phrase_generator, new_node_embedding, target_id)
                    phrase_output.append(p_output)
                    target_output.append(tokenizer.decode(y_tokenize.view(-1).cpu().numpy()))
            for i in range(len(phrase_output)):
                total += 1
                if phrase_output[i] == target_output[i]:
                    count += 1
                print("{} - {} / {}".format(index2phrase[X[j][i]], phrase_output[i], target_output[i]))
            #print(phrase_output)
            #print(target_output)
            print("-" * 100)
        print("ACC:{:.4f}".format(count/total))