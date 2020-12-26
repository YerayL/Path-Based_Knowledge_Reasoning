import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from transformers import BertForSequenceClassification, BertModel


class Classifier(nn.Module):
    def __init__(self, encoder, out_class, hidden_size=256, dropout_p=0.1):
        super().__init__()
        out_dim = encoder[1].out_dim
        self.query_emb = nn.Embedding(num_embeddings=17, embedding_dim=250, padding_idx=None)
        self.cnn_encoder = encoder[0]
        self.encoder = encoder[1]
        self.out = nn.Sequential(
                        nn.Linear(out_dim, out_dim),
                        nn.Tanh())

    def forward(self, x, padding_to2, query):
        # x_query = x[:, 0, :, :]
        # x_query = x_query[:,0,:]

        x = self.encoder(x, padding_to2)
        query = torch.unsqueeze(self.query_emb(query), dim=1)
        # query = torch.unsqueeze(self.cnn_encoder(x_query), dim=1)

        # query = torch.unsqueeze(x[:, 0], dim=1)  # use bert encode query

        # path = x[:, 1:]
        path = x
        a = F.softmax(torch.mul(self.out(path), query).sum(2))
        a = torch.unsqueeze(a, dim=2)
        context = torch.mul(a, path).sum(1)
        score1 = torch.sum(torch.mul(nn.Tanh()(context), torch.squeeze(query)), dim=1, keepdim=True).float()
        inv_score = score1 - score1
        _score = torch.cat([inv_score, score1], dim=1)
        p = F.softmax(_score)
        score = F.log_softmax(_score)

        return p, score

class BertClassification(nn.Module):
    def __init__(self, model="bert-base-uncased"):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model)

    def forward(self, x, x_length=[]):
        output = self.bert(x)
        return output[0]

class BertEncoder(nn.Module):
    def __init__(self, out_dim, pretrained_model="bert-base-uncased", dropout_p=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.bert = BertModel.from_pretrained(pretrained_model)

        # for name, param in self.bert.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        # for p in self.bert.parameters():
        #    p.requires_grad = False

        self.out = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.bert.config.hidden_size, out_dim))


    def forward(self, x, padding_to2):

        x_input = x[:, 0, :,:]
        x_attn = x[:, 1, :,:]

        batch_size = x_input.size(0)
        path_num = x_input.size(1)
        length = x_input.size(2)
        x_input = x_input.reshape(-1, length)
        x_attn = x_attn.reshape(-1, length)
        x, _ = self.bert(x_input, attention_mask=x_attn)
        x = nn.ReLU()(self.out(x[:, 0]))
        o = x.view(batch_size, path_num, -1)
        # for i in range(padding_to2):
        #     x, _ = self.bert(x_input[:,i,:], attention_mask=x_attn[:,i,:])
        #     x = nn.ReLU()(self.out(x[:, 0]))
        #     if i == 0:
        #         o = x
        #     elif i == 1:
        #         o = torch.stack([o, x], dim=1)
        #     else:
        #         x = torch.unsqueeze(x, dim=1)
        #         o = torch.cat([o, x], dim=1)

        return o



