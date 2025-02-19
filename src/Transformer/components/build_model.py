import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from Transformer import logger
from Transformer.entity.config_entity import BuildModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Initializes the TokenEmbedding module.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimensionality of the input and output embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len)

        Returns:
            torch.Tensor: embedded input tensor (batch_size, seq_len, d_model)
        """
        # Multiply embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length, dropout=0.1):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): Dimensionality of the input embeddings.
            max_length (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix exactly as described in the paper
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of the positional encoding layer.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: output tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V, and output
        self.w_qkv = nn.Linear(d_model, d_model * 3)
        self.w_o = nn.Linear(d_model, d_model)  # Added missing output projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass of the Multi-Head Attention module.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): attention mask tensor. Defaults to None.

        Returns:
            tuple: output tensor with attention applied (batch_size, seq_len, d_model), attention weights
        """
        batch_size = x.size(0)

        # Linear projections for Q, K, V at once
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # (B, H, Sq, d_k)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # (B, H, Sk, d_k)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # (B, H, Sv, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (B, H, Sq, Sk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropout = self.dropout(attn_weights)

        output = torch.matmul(attn_weights_dropout, v)  # (B, H, Sq, d_k)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # (B, Sq, d_model)

        return self.w_o(output), attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initializes the PositionwiseFeedForward module.

        Args:
            d_model (int): Dimensionality of the input and output embeddings.
            d_ff (int): Dimensionality of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the PositionwiseFeedForward module.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: output tensor (batch_size, seq_len, d_model)
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initializes an Encoder Layer.

        Args:
            d_model (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass of the EncoderLayer.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): attention mask. Defaults to None.

        Returns:
            tuple: output tensor (batch_size, seq_len, d_model), attention weights
        """
        # Self-Attention with residual connection and layer normalization
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Position-wise Feed-Forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initializes a Decoder Layer.

        Args:
            d_model (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of the DecoderLayer.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len, d_model)
            enc_output (torch.Tensor): encoder output tensor (batch_size, seq_len, d_model)
            src_mask (torch.Tensor, optional): source mask for cross attention. Defaults to None.
            tgt_mask (torch.Tensor, optional): target mask for self attention. Defaults to None.

        Returns:
            tuple: output tensor (batch_size, seq_len, d_model), self attention weights, cross attention weights
        """
        # Masked Self-Attention with residual connection and layer normalization
        attn_output, self_attn_weights = self.self_attn(x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Cross-Attention with encoder outputs
        attn_output, cross_attn_weights = self.cross_attn(x, mask=src_mask)
        x = self.norm2(x + self.dropout2(attn_output))

        # Position-wise Feed-Forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, self_attn_weights, cross_attn_weights


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1
    ):
        """
        Initializes the Encoder module.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward network.
            num_layers (int): Number of layers in the encoder.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len)
            mask (torch.Tensor, optional): attention mask (batch_size, 1, 1, seq_len). Defaults to None.

        Returns:
            tuple: output tensor (batch_size, seq_len, d_model), list of attention weights
        """
        # Token embedding and positional encoding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        attention_weights = []

        # Pass through encoder layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)

        return self.norm(x), attention_weights


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1
    ):
        """
        Initializes the Decoder module.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward network.
            num_layers (int): Number of layers in the decoder.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): input tensor (batch_size, seq_len)
            enc_output (torch.Tensor): encoder output tensor (batch_size, seq_len, d_model)
            src_mask (torch.Tensor, optional): source mask. Defaults to None.
            tgt_mask (torch.Tensor, optional): target mask. Defaults to None.

        Returns:
            tuple: output tensor (batch_size, seq_len, vocab_size), self attention weights, cross attention weights
        """
        # Token embedding and positional encoding
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        self_attention_weights = []
        cross_attention_weights = []

        # Pass through decoder layers
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)

        x = self.norm(x)
        return self.fc_out(x), self_attention_weights, cross_attention_weights


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len,
        dropout,
    ):
        """
        Initializes the Transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward network.
            num_layers (int): Number of layers in the encoder and decoder.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout
        )
        self.decoder = Decoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout
        )
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): source tensor (batch_size, src_seq_len)
            tgt (torch.Tensor): target tensor (batch_size, tgt_seq_len)
            src_mask (torch.Tensor, optional): source mask. Defaults to None.
            tgt_mask (torch.Tensor, optional): target mask. Defaults to None.

        Returns:
            tuple: output tensor and attention weights dictionary
        """
        enc_output, enc_attentions = self.encoder(src, src_mask)
        dec_output, dec_self_attentions, dec_cross_attentions = self.decoder(
            tgt, enc_output, src_mask, tgt_mask
        )

        # Collect all attention weights
        attention_weights = {
            "encoder_attention": enc_attentions,
            "decoder_self_attention": dec_self_attentions,
            "decoder_cross_attention": dec_cross_attentions,
        }

        return dec_output, attention_weights

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """
        Generate a square mask for the sequence.

        Args:
            sz (int): Sequence length.

        Returns:
            torch.Tensor: masked filled with zeros for future positions and ones for current/past positions
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class BuildModel:
    def __init__(self, config: BuildModelConfig):
        """
        This method initializes the build model component by
        setting the configuration.

        :param config: BuildModelConfig entity
        """
        self.config = config

    def build(self):
        """
        Constructs the Transformer model using the configuration details.

        This method initializes the Transformer model with parameters such as
        the number of layers, model dimensions, number of attention heads,
        feed-forward network dimensions, vocabulary size, dropout rate, and
        maximum sequence length.

        :return: None
        """
        logger.info("Building the model...")
        self.transformer = Transformer(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            d_ff=self.config.dff,
            num_layers=self.config.num_layers,
            max_len=self.config.max_length,
            dropout=self.config.dropout,
        )

    def save_model(self):
        """
        Saves the model to a file.

        :return: None
        """
        torch.save(self.transformer.state_dict(), self.config.model_path)
        logger.info(f"Model saved at: {self.config.model_path}")
