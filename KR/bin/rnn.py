#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/8 14:01
# @Author  : Yinyu Lan

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class RNNCell_rel(nn.Module):
    def __init__(self, e_input_size, r_input_size, hidden_size):
        super(RNNCell_rel, self).__init__()
        self.e_input_size = e_input_size
        self.r_input_size = r_input_size
        self.hidden_size = hidden_size
        self.e2h = nn.Linear(e_input_size, hidden_size)
        self.r2h = nn.Linear(r_input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, e, r, hidden):
        '''

        :param x: [batch_size, input_size]
        :param hidden: [batch_size, hidden_size]
        :return: h_n: [batch_size, hidden_size]
        '''
        return torch.relu(self.e2h(e) + self.r2h(r) + self.h2h(hidden))


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        '''

        :param x: [batch_size, input_size]
        :param hidden: [batch_size, hidden_size]
        :return: h_n: [batch_size, hidden_size]
        '''
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        h_n = newgate + inputgate * (hidden - newgate)
        return h_n


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        '''

        :param x: [batch_size, input_size]
        :param hidden: tuple of [batch_size, hidden_size]
        :return: (h_n, c_n), each size is [batch_size, hidden_size]
        '''
        hx, cx = hidden
        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c_n = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        h_n = torch.mul(outgate, torch.tanh(c_n))
        return (h_n, c_n)


class RNNModel(nn.Module):
    def __init__(self, e_input_size=100, r_input_size=250, hidden_size=250, num_layers=1, bidirectional=False, batch_first=True, dropout=0,
                 mode="RNN"):
        super(RNNModel, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions = 2 if bidirectional else 1
        self.mode = mode
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.cells = cells = nn.ModuleList()


        if mode == "RNN":
            cell_cls = RNNCell_rel
        elif mode == "GRU":
            cell_cls = GRUCell
        elif mode == "LSTM":
            cell_cls = LSTMCell
        else:
            raise NotImplementedError(mode + " mode not supported, choose 'RNN', 'GRU' or 'LSTM'.")

        for layer in range(num_layers):
            for direction in range(num_directions):
                rnn_cell = cell_cls(e_input_size, r_input_size, hidden_size) if layer == 0 else cell_cls(hidden_size * num_directions,
                                                                                         hidden_size)
                cells.append(rnn_cell)

    def forward(self, entity, relation):
        '''

        :param x: [batch_size, max_seq_length, input_size] if batch_first is True
        :return: output: [batch, seq_len, num_directions * hidden_size] if batch_first is True
                 hidden: [num_layers * num_directions, batch, hidden_size] if mode is "RNN" or "GRU", if mode is "LSTM",
                         hidden will be (h_n, c_n), each size is [num_layers * num_directions, batch, hidden_size].
        '''
        if self.batch_first:
            batch_size = entity.size(0)
            inputs_ent = entity.transpose(0, 2)
            inputs_rel = relation.transpose(0, 2)
        else:
            batch_size = entity.size(1)
            inputs_ent = entity
            inputs_rel = relation

        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(entity.device)
        if self.mode == 'LSTM':
            h0 = (h0, h0)
        outs = []
        hiddens = []
        for layer in range(self.num_layers):
            # inputs = inputs if layer == 0 else self.dropout(outs)  # [max_seq_length, batch_size, layer_input_size]
            layer_outs_with_directions = []
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                # inputs = inputs if direction == 0 else inputs.flip(0)
                rnn_cell = self.cells[idx]
                if self.mode == 'LSTM':
                    layer_hn = (h0[0][idx], h0[1][idx])  # tuple of [batch_size, hidden_size], (h0, c0)
                else:
                    layer_hn = h0[idx]
                layer_outs = []
                for time_step in range(8):
                    layer_hn = rnn_cell(inputs_ent[time_step],inputs_rel[time_step], layer_hn)
                    layer_outs.append(layer_hn)
                if self.mode == 'LSTM':
                    layer_outs = torch.stack([out[0] for out in layer_outs])  # [max_seq_len, batch_size, hidden_size]
                else:
                    layer_outs = torch.stack(layer_outs)  # [max_seq_len, batch_size, hidden_size]
                layer_outs_with_directions.append(layer_outs if direction == 0 else layer_outs.flip(0))
                hiddens.append(layer_hn)
            outs = torch.cat(layer_outs_with_directions, -1)  # [max_seq_len, batch_size, 2*hidden_size]

        if self.batch_first:
            output = outs.transpose(0, 2)
        else:
            output = outs
        if self.mode == 'LSTM':
            hidden = (torch.stack([h[0] for h in hiddens]), torch.stack([h[1] for h in hiddens]))
        else:
            hidden = torch.stack(hiddens)
        need = hidden[0].transpose(0, 1)
        return output, need


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # out_dim = encoder.hidden_size
        self.entity_emb = nn.Embedding(num_embeddings=2218, embedding_dim=100, padding_idx=2217)
        self.relation_emb = nn.Embedding(num_embeddings=51390, embedding_dim=250, padding_idx=51389)
        self.query_emb = nn.Embedding(num_embeddings=46, embedding_dim=250, padding_idx=None)
        self.encoder = encoder
        # self.out = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.Tanh())

    def forward(self, entity, relation, query, path_num):
        entity_types = self.entity_emb(entity)     # [batch_size,path_num,path_length,entity_types_num,emb_dim]
        relation = self.relation_emb(relation)
        query = self.query_emb(query)   # [1,emb_dim]
        entity = entity_types.sum(3)/7    # [batch_size,path_num,path_length,emb_dim]
        # for i in range(path_num):
        #     _, x = self.encoder(entity[:, i], relation[:, i])
        #     if i == 0:
        #         path = x
        #     elif i == 1:
        #         path = torch.stack([path, x], dim=1)
        #     else:
        #         x = torch.unsqueeze(x, dim=1)
        #         path = torch.cat([path, x], dim=1)
        _, path = self.encoder(entity, relation)

        # path:[batch_size,path_num,hid_dim] query: [1,250]
        query = torch.unsqueeze(query, dim=1)
        a = F.softmax(torch.mul(path, query).sum(2))
        a = torch.unsqueeze(a, dim=2)
        context = torch.mul(a, path).sum(1)
        score1 = torch.sum(torch.mul(nn.Tanh()(context), torch.squeeze(query)), dim=1, keepdim=True).float()
        inv_score = score1 - score1
        _score = torch.cat([inv_score, score1], dim=1)
        p = F.softmax(_score)
        score = F.log_softmax(_score)
        return p, score




def test_RNN_Model():
    input_size, hidden_size, num_layers, bidirectional, batch_first = 50, 100, 2, True, False
    dropout = 0.1

    x = torch.randn([20, 15, 50])
    mymodel = RNNModel(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first,
                       dropout=dropout,
                       mode="RNN")
    for cell in mymodel.cells:
        for w in cell.parameters():
            nn.init.constant_(w.data, 0.01)

    model = nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first,
                   dropout=dropout)
    for w in model.parameters():
        nn.init.constant_(w.data, 0.01)

    torch.manual_seed(1)
    outs, hidden = model(x)
    torch.manual_seed(1)
    myouts, myhidden = mymodel(x)

    assert (hidden != myhidden).sum().item() == 0, "hidden don't match, RNNcell maybe wrong!"
    assert (outs != myouts).sum().item() == 0, "outs don't match, RNNcell maybe wrong!"


def test_GRU_Model():
    input_size, hidden_size, num_layers, bidirectional, batch_first = 50, 100, 2, True, False
    dropout = 0.1
    torch.manual_seed(1)
    x = torch.randn([20, 15, 50])

    mymodel = RNNModel(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True,
                       dropout=dropout,
                       mode="GRU")
    for cell in mymodel.cells:
        for w in cell.parameters():
            nn.init.constant_(w.data, 0.01)

    model = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
    for w in model.parameters():
        nn.init.constant_(w.data, 0.01)

    torch.manual_seed(1)
    outs, hidden = model(x)
    torch.manual_seed(1)
    myouts, myhidden = mymodel(x)

    assert (hidden != myhidden).sum().item() == 0, "hidden don't match, GRUcell maybe wrong!"
    assert (outs != myouts).sum().item() == 0, "outs don't match, GRUcell maybe wrong!"


def test_LSTM_Model():
    input_size, hidden_size, num_layers, bidirectional, batch_first = 50, 100, 2, True, False
    dropout = 0.1

    x = torch.randn([20, 15, 50])

    mymodel = RNNModel(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first,
                       dropout=dropout,
                       mode="LSTM")
    for cell in mymodel.cells:
        for w in cell.parameters():
            nn.init.constant_(w.data, 0.01)

    model = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first,
                    dropout=dropout)
    for w in model.parameters():
        nn.init.constant_(w.data, 0.01)

    torch.manual_seed(1)
    outs, (h_n, c_n) = model(x)
    torch.manual_seed(1)
    myouts, (myh_n, myc_n) = mymodel(x)

    assert (h_n != myh_n).sum().item() == 0, "h_n don't match, LSTMcell maybe wrong!"
    assert (c_n != myc_n).sum().item() == 0, "c_n don't match, LSTMcell maybe wrong!"
    assert (outs != myouts).sum().item() == 0, "outs don't match, LSTMcell maybe wrong!"


def test():
    test_RNN_Model()
    test_GRU_Model()
    test_LSTM_Model()
