import torch
import torch.nn as nn
import numpy as np
import math  # Add this line
from transformers import BertModel, RobertaModel, XLNetModel, DistilBertModel, AlbertModel  #newly Added

from common import get_parser



parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



#Adding Attention Layer


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


    def transpose_for_scores(self, x, seq_len):
        """Reshape x for attention scores calculation."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add an extra dimension if input has only two dimensions
        batch_size, hidden_size = x.size()
        new_x_shape = (batch_size, self.num_attention_heads, self.attention_head_size, seq_len)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # Permute to achieve (batch_size, num_attention_heads, attention_head_size, seq_len)


    
    '''
    def transpose_for_scores(self, x):
        # Print the actual input shape before reshaping
        print("Input shape before reshape:", x.shape)
        # ... rest of your code calculating new_x_shape
        new_x_shape = (x.size(0), self.num_attention_heads, self.attention_head_size, x.size(1) // (self.num_attention_heads * self.attention_head_size))
        # Print the calculated expected shape
        print("Expected reshaped shape:", new_x_shape)
        
        # ... rest of your code
        
        # Print the actual shape after reshape (causing the error)
        x = x.view(*new_x_shape)
        print("Actual reshaped shape:", x.shape)
        return x.permute(0, 2, 1, 3)
    '''

    ''' 
    def transpose_for_scores(self, x):
        #print("Input shape:", x.shape)
        #new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        new_x_shape = (x.size(0), self.num_attention_heads, self.attention_head_size) + x.size()[1:]
        print("Expected reshaped shape:", new_x_shape)
        x = x.view(*new_x_shape)
        #print("Actual reshaped shape:", x.shape)
        return x.permute(0, 2, 1, 3)
    '''


    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
    
        # Print out the shape of the mixed query, key, and value layers
        print("Mixed Query Layer shape:", mixed_query_layer.shape)
        print("Mixed Key Layer shape:", mixed_key_layer.shape)
        print("Mixed Value Layer shape:", mixed_value_layer.shape)
    
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
    
        # Print out the shape after transpose_for_scores
        print("Query Layer shape after transpose_for_scores:", query_layer.shape)
        print("Key Layer shape after transpose_for_scores:", key_layer.shape)
        print("Value Layer shape after transpose_for_scores:", value_layer.shape)
    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
    
        # Print out the shape after attention calculation
        print("Attention Scores shape:", attention_scores.shape)
    
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
    
        # Print out the shape after softmax operation
        print("Attention Probs shape:", attention_probs.shape)


    
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # After context_layer = torch.matmul(attention_probs, value_layer)
        new_context_layer_shape = (context_layer.size(0), self.all_head_size) + context_layer.size()[1:]
        context_layer = context_layer.view(*new_context_layer_shape)

        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #context_layer = context_layer.view(*new_context_layer_shape)
    
        # Print out the shape after context layer calculation
        print("Context Layer shape:", context_layer.shape)
    
        return context_layer






#Adding to debug but not working

'''
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
        print("Input shape:", x.shape)
        # Assuming x has shape (batch_size, seq_len, hidden_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, hidden_size = x.size()
        new_x_shape = (batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        print("Reshaped shape:", x.shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        

        # Rest of the forward method...


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
'''





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
    


class AlbertFGBC(nn.Module):
    def __init__(self, pretrained_model=args.pretrained_model):
        super().__init__()
        self.Albert = AlbertModel.from_pretrained(pretrained_model)
        self.attention = SelfAttentionLayer(args.albert_hidden, args.num_attention_heads, args.attention_dropout)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.albert_hidden, 64)  # Adjust the input size for linear layer
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    
    def forward(self, input_ids, attention_mask):
        _, last_hidden_state = self.Albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
            )
            seq_len = last_hidden_state.size(1)  # Get sequence length from hidden state
            
            # Apply self-attention with sequence length
        attention_output = self.attention(last_hidden_state, attention_mask, seq_len)
            # ... rest of your code
            print(f'Self-Attention Output Shape: {attention_output.shape}')
    
    
    #def forward(self, input_ids, attention_mask):
       # _, last_hidden_state = self.Albert(
       #     input_ids=input_ids,
        #    attention_mask=attention_mask,
         #   return_dict=False
        #)
        #print(f'Albert Last Hidden State Shape: {last_hidden_state.shape}')

        # Apply self-attention
       # attention_output = self.attention(last_hidden_state, attention_mask)
        

        # Process attention output
        bo = self.drop1(attention_output)
        bo = self.linear(bo)
        print(f'Linear Output Shape: {bo.shape}')
        
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)
        print(f'Final Output Shape: {bo.shape}')

        output = self.out(bo)
        print(f'Output Shape: {output.shape}')

        return output




'''commenting off this below block for debugging

class AlbertFGBC(nn.Module): # Below lines newly added

    def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.Albert = AlbertModel.from_pretrained(pretrained_model)
        self.attention = SelfAttentionLayer(args.albert_hidden, args.num_attention_heads, args.attention_dropout)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.albert_hidden, 64)  # Adjust the input size for linear layer
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        _, last_hidden_state = self.Albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        # Apply self-attention
        attention_output = self.attention(last_hidden_state, attention_mask)

        # Process attention output
        bo = self.drop1(attention_output)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output 
        #upto this new code for attention layer


'''


    #below is old code by T 
''' Commenting off for attention test
def __init__(self, pretrained_model = args.pretrained_model):
        super().__init__()
        self.Albert = AlbertModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.roberta_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        _,last_hidden_state = self.Albert(
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
