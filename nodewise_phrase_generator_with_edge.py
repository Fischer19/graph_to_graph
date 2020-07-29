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
from transformers import BertTokenizer
import argparse
import time


def gen_phrase_with_edge(phrase_decoder, new_node_embedding, target_id, edge_label, teacher_forcing = True):
    loss = 0
    criterion = nn.CrossEntropyLoss()
    index = target_id
    y_tokenize = torch.Tensor(tokenizer(index2phrase[index])['input_ids'][1:]).view(batch_size,-1).long().to(device)
    output = []
    phrase_decoder_input = torch.tensor([CLS_token], device=device).view(1, 1)
    phrase_decoder_hidden = new_node_embedding
    for ni in range(y_tokenize.shape[1] + 1):
        phrase_decoder_output, node_decoder_hidden = phrase_generator(
            phrase_decoder_input, phrase_decoder_hidden, ni == 0)
        topv, topi = phrase_decoder_output.topk(1)
        if teacher_forcing:
            if ni != 0:
                phrase_decoder_input = y_tokenize[:,ni-1].view(batch_size, 1)
            else:
                phrase_decoder_input = torch.Tensor([edge_label]).view(1,1).long().to(device)
        if ni != 0:
            loss += criterion(phrase_decoder_output.view(batch_size,-1), y_tokenize[:, ni-1].view(batch_size))
        else:
            loss += criterion(phrase_decoder_output.view(batch_size,-1), torch.Tensor([edge_label]).long().to(device))
        output.append(topi.squeeze().detach().cpu().numpy())

    return loss, tokenizer.decode(output[1:]), y_tokenize, rel_name[edge_label], rel_name[output[0]]

def train(model_list, g_list, args, teacher_forcing = True):
    #CUDA_LAUNCH_BLOCKING=1
    start_time = time.time()
    graph_encoder, node_generator, phrase_generator = model_list
    learning_rate = args.learning_rate
    graph_encoder_optimizer = optim.Adagrad(graph_encoder.parameters(), lr=learning_rate)
    node_decoder_optimizer = optim.Adagrad(node_generator.parameters(), lr=learning_rate)
    phrase_decoder_optimizer = optim.Adagrad(phrase_generator.parameters(), lr=learning_rate)

    for i in range(args.iteration):
        for j in range(args.train_num):
            node_decoder_optimizer.zero_grad()
            graph_encoder_optimizer.zero_grad()
            phrase_decoder_optimizer.zero_grad()
            
            node_embedding, g_embedding = graph_encoder(g_list[j])
            output = []
            phrase_output = []
            criterion = nn.CrossEntropyLoss()
            phrase_loss_sum = 0
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
                    phrase_loss, p_output, _, edge_name, _ = gen_phrase_with_edge(phrase_generator, new_node_embedding, target_id, edge_label[j][k])
                    phrase_output.append(edge_name + ' ' + p_output)
                    loss += phrase_loss
                    phrase_loss_sum += phrase_loss

            loss.backward(retain_graph = True)
            node_decoder_optimizer.step()
            graph_encoder_optimizer.step()
            phrase_decoder_optimizer.step()
        if i % 10 == 0:
            elapsed_time = time.time() - start_time
            print("Phrase loss: {:.4f} / Total loss: {:.4f}--Time elapsed{}".format(phrase_loss_sum.cpu().item(), loss.cpu().item(), time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
            torch.save(node_generator.state_dict(), "parameters/nodewise_node_decoder_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size))
            torch.save(graph_encoder.state_dict(), "parameters/nodewise_graph_encoder_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size))
            torch.save(phrase_generator.state_dict(), "parameters/nodewise_phrase_generator_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size))

def evaluate(nums, g_list, model_list):
    graph_encoder, node_generator, phrase_generator = model_list
    count = 0
    count_edge = 0
    total = 0
    teacher_forcing = True
    for j in range(nums):
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
                phrase_loss, p_output, y_tokenize, edge_name, predicted_edge = gen_phrase_with_edge(phrase_generator, new_node_embedding, target_id, edge_label[j][k])
                phrase_output.append(edge_name + ' ' + p_output)
                target_output.append(predicted_edge + ' ' + tokenizer.decode(y_tokenize.view(-1).cpu().numpy()))
        for i in range(len(phrase_output)):
            total += 1
            if phrase_output[i].split(' ')[1:] == target_output[i].split(' ')[1:]:
                count += 1
            if phrase_output[i].split(' ')[0] == target_output[i].split(' ')[0]:
                count_edge += 1
            print("{} - {} / {}".format(index2phrase[X[j][i]], phrase_output[i], target_output[i]))
        #print(phrase_output)
        #print(target_output)
        print("-" * 100)
    print("Node ACC:{:.4f}".format(count/total))
    print("Edge ACC:{:.4f}".format(count_edge/total))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--train', action = "store_true")
    parser.add_argument('-r', '--resume', action = "store_true")
    parser.add_argument('-hd', '--hidden_size', required = False, type = int, default = 64)
    parser.add_argument('-id', '--input_size', required = False, type = int, default = 300)
    parser.add_argument('-e', '--embedding', required = False, default = "conceptnet")
    parser.add_argument('-i', '--iteration', required = False, type = int, default = 200)
    parser.add_argument('-n', '--train_num', required = False, type = int, default = 200)
    parser.add_argument('-lr', '--learning_rate', required = False, type = float, default = 2e-3)
    args = parser.parse_args()

    # Load knowledge graph structural information:
    with open("DGL_graph.pkl", "rb") as f:
        g = pickle.load(f)

    if args.embedding == "BERT":
        embedding_values = torch.load("data/conceptnet/bert_embedding_pooled.pt")
    else:
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

    #X,y = gen_training_set(sub_graph_G, 3, 1000, nodewise=True)
    #Load training data of size 1000 
    with open("nodewise_training_data.pkl", "rb") as f:
        dic = pickle.load(f)
        X = dic["X"]
        y = dic["y"]
        edge_label = dic["edge"]
    rel_name = ['DefinedAs', 'HasPainIntensity', 'Causes', 'HasLastSubevent', 
    'UsedFor', 'RelatedTo', 'NotMadeOf', 'HasProperty', 'DesireOf', 'NotHasA', 
    'ReceivesAction', 'NotHasProperty', 'Desires', 'HasSubevent', 'HasA', 'NotDesires', 
    'NotIsA', 'InstanceOf', 'MotivatedByGoal', 'LocatedNear', 'SymbolOf', 'CreatedBy', 
    'AtLocation', 'InheritsFrom', 'CapableOf', 'HasFirstSubevent', 'LocationOfAction', 
    'PartOf', 'IsA', 'NotCapableOf', 'MadeOf', 'HasPrerequisite', 'CausesDesire', 'HasPainCharacter']

    # Define tokenizer using BertTokenizer
    tokenizer = BertTokenizer("data/conceptnet/embedding_keys.txt")
    # Load KGB Corpus
    with open("data/conceptnet/embedding_keys.pkl", "rb") as f:
        embedding_keys = pickle.load(f)
    for i,item in enumerate(embedding_keys):
        words = item.replace("_", " ")
        embedding_keys[i] = words
    embedding_keys = np.array(embedding_keys)
    index2phrase = embedding_keys[component]

    # Instantiate models
    CORPUS_SIZE = len(tokenizer.vocab)
    input_size = args.input_size
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
    phrase_generator = LSTM_phrase_generator(hidden_size * 2, phrase_output_size, 34)

    #node_generator.load_state_dict(torch.load("parameters/nodewise_node_decoder_parameters_BERT_joint.pth"))
    #graph_encoder.load_state_dict(torch.load("parameters/nodewise_graph_encoder_parameters_BERT_joint.pth"))
    #phrase_generator.load_state_dict(torch.load("parameters/nodewise_phrase_generator_parameters_BERT_joint.pth"))

    graph_encoder.to(device)
    node_generator.to(device)
    phrase_generator.to(device)

    # Training preperations
    CLS_token = 0
    SEP_token = 1

    g_list = []
    y_list = []
    for i in range(args.train_num):
        g = sub_graph_G.subgraph(X[i])
        g.copy_from_parent()
        g.ndata["x"] = g.ndata["x"].float().to(device)
        edge_norm = torch.ones(g.edata['rel_type'].shape[0]).to(device)
        g.edata.update({'norm': edge_norm.view(-1,1).to(device)})
        g_list.append(g)
        y_list.append(y[i])

    print("-" * 10, "Start Training", "-" * 10)
    if args.train:
        if args.resume:
            node_generator.load_state_dict(torch.load("parameters/nodewise_node_decoder_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size)))
            graph_encoder.load_state_dict(torch.load("parameters/nodewise_graph_encoder_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size)))
            phrase_generator.load_state_dict(torch.load("parameters/nodewise_phrase_generator_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size)))
        train([graph_encoder, node_generator, phrase_generator], g_list, args)
    else:
        node_generator.load_state_dict(torch.load("parameters/nodewise_node_decoder_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size)))
        graph_encoder.load_state_dict(torch.load("parameters/nodewise_graph_encoder_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size)))
        phrase_generator.load_state_dict(torch.load("parameters/nodewise_phrase_generator_parameters_input{}_h{}.pth".format(args.input_size, args.hidden_size)))
        evaluate(args.train_num, g_list, [graph_encoder, node_generator, phrase_generator])
