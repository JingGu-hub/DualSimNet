
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# define Encoder
class Transformer(nn.Module):
    def __init__(self, input_dim, input_length, embedding_size, feature_size, num_classes, num_layers=2, num_heads=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = nn.Linear(input_length, embedding_size)
        self.time_attention = nn.TransformerEncoderLayer(batch_first=True, d_model=embedding_size, nhead=num_heads)
        self.norm1 = nn.LayerNorm([input_dim, embedding_size])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(input_dim * embedding_size, num_classes)

        self.decoder = nn.Sequential(
            nn.Linear(input_dim * embedding_size, feature_size),
            nn.Linear(feature_size, (input_dim * input_length) // 2),
            nn.Linear((input_dim * input_length) // 2, input_dim * input_length)
        )

    def forward(self, x, task_type):
        x = self.embedding(x)
        for attn_layer in range(self.num_layers):
            x = self.time_attention(x)
            x = F.relu(x)
            x = self.norm1(x)
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        if task_type == 'classification':
            features = x
            classification_output = self.projection(features)
            return classification_output, features
        elif task_type == 'restruction':
            features = x
            restruction_outputs = self.decoder(features)
            restruction_outputs = restruction_outputs.reshape(restruction_outputs.size(0), self.input_dim, self.input_length)
            return restruction_outputs, features

