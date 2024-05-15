import torch
import torch.nn as nn
import numpy as np
import math
from transformers import BertModel, RobertaModel, XLNetModel, DistilBertModel, AlbertModel
from common import get_parser

# Parsing arguments from command line
parser = get_parser()
args = parser.parse_args()

# Setting random seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


#Adding Self-attention layer to compute attention scores while applying them to the input embeddings
class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout):
        super(SelfAttentionLayer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Adjust attention mask shape
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class AlbertFGBC(nn.Module):
    def __init__(self, pretrained_model=args.pretrained_model):
        super().__init__()
        self.Albert = AlbertModel.from_pretrained(pretrained_model)
        self.attention = SelfAttentionLayer(args.albert_hidden, args.num_attention_heads, args.attention_dropout)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.albert_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.Albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        last_hidden_state = outputs[0]

        # Apply self-attention
        attention_output = self.attention(last_hidden_state, attention_mask)

        # Mean pooling over the sequence dimension
        attention_output = attention_output.mean(dim=1)

        # Process attention output
        bo = self.drop1(attention_output)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = torch.tanh(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

class BertFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.Bert = BertModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.bert_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _,last_hidden_state = self.Bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        #print(f'Last Hidden State - {last_hidden_state.shape}')
        bo = self.drop1(last_hidden_state)
        #print(f'Dropout1 - {bo.shape}')
        bo = self.linear(bo)
        #print(f'Linear1 - {bo.shape}')
        bo = self.batch_norm(bo)
        #print(f'BatchNorm - {bo.shape}')
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)
        #print(f'Dropout2 - {bo.shape}')

        output = self.out(bo)
        #print(f'Output - {output.shape}')
        return output

class RobertaFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.Roberta = RobertaModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.roberta_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        _,last_hidden_state = self.Roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        bo = self.drop1(last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

'''
class DistilBertFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.DistilBert = DistilBertModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.distilbert_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.DistilBert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        
        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state
'''

class DistilBertFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.DistilBert = DistilBertModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.distilbert_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.DistilBert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        
        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state


class XLNetFGBC(nn.Module):
    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.XLNet = XLNetModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.xlnet_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state = self.XLNet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)

        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output
        
    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

