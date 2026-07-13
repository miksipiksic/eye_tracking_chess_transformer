"""
transformer_model.py — Encoder-Decoder Transformer 

Adaptations:
  - Vocab: 66 tokens
  - Encoder input: squares noticed [e4, d5, e5, ...]
  - Decoder input: [SOS] token + autoreg next move prediction
  - Output: ProjectionLayer → 64 logits (1 per square)
  - Greedy decode: decoder generates 1 token = predicted square (top 1)

"""

import torch
import torch.nn as nn
import math

PAD_IDX   = 64
SOS_IDX   = 65
VOCAB_SIZE = 66   # 64  + PAD + SOS 
N_SQUARES  = 64   # output classes


# target vocab size = 66 (like source)
TARGET_VOCAB_SIZE = 66



def square_to_idx(sq: str) -> int:
    sq   = sq.strip()
    file = ord(sq[0]) - ord('a')
    rank = int(sq[1]) - 1
    return rank * 8 + file


def idx_to_square(idx: int) -> str:
    return f"{chr(ord('a') + idx % 8)}{idx // 8 + 1}"


def causal_mask(size: int) -> torch.Tensor:
    """Decoder can't see future tokens."""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class InputEmbeddings(nn.Module):
    """Embeddings"""
    def __init__(self, model_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size       = vocab_size
        self.embedding        = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.model_dimension)





class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int,
                 context_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(context_size, model_dimension)
        position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2).float() *
            (-math.log(10000.0) / model_dimension)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps   = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias  = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, model_dimension: int,
                 feed_forward_dimension: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dimension, model_dimension)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, model_dimension: int,
                 heads: int, dropout: float) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.heads           = heads
        self.dropout         = nn.Dropout(dropout)

        assert model_dimension % heads == 0, \
            "model_dimension moduo heads must be 0 ."

        self.head_dimension = model_dimension // heads
        self.w_q = nn.Linear(model_dimension, model_dimension)
        self.w_k = nn.Linear(model_dimension, model_dimension)
        self.w_v = nn.Linear(model_dimension, model_dimension)
        self.w_o = nn.Linear(model_dimension, model_dimension)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        head_dimension    = query.shape[-1]
        attention_scores  = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)

        B = query.shape[0]
        query = query.view(B, -1, self.heads, self.head_dimension).transpose(1, 2)
        key   = key.view(B, -1, self.heads, self.head_dimension).transpose(1, 2)
        value = value.view(B, -1, self.heads, self.head_dimension).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(B, -1, self.heads * self.head_dimension)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block   = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block  = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block    = feed_forward_block
        self.residual_connections  = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, source_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(features)

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, model_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, x):
        return self.proj(x)  # logits


class Transformer(nn.Module):
    def __init__(self, encoder, decoder,
                 source_embed, target_embed,
                 source_pos, target_pos,
                 projection_layer) -> None:
        super().__init__()
        self.encoder          = encoder
        self.decoder          = decoder
        self.source_embed     = source_embed
        self.target_embed     = target_embed
        self.source_pos       = source_pos
        self.target_pos       = target_pos
        self.projection_layer = projection_layer

    def encode(self, source, source_mask):
        source = self.source_embed(source)
        source = self.source_pos(source)
        return self.encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)




def build_transformer(
        source_vocab_size: int  = VOCAB_SIZE,  # unused, kept for compatibility
        target_vocab_size: int  = TARGET_VOCAB_SIZE,
        source_context_size: int = 20,
        target_context_size: int = 2,
        model_dimension: int    = 128,
        number_of_blocks: int   = 3,
        heads: int              = 4,
        dropout: float          = 0.1,
        feed_forward_dimension: int = 256
) -> Transformer:
    """
    source = squares seq  (vocab: 66 tokena, context: 20)
    target = [SOS] + predicted (vocab: 64 squares,  context: 2)
    """
    source_embed = InputEmbeddings(model_dimension, source_vocab_size)
    target_embed = InputEmbeddings(model_dimension, target_vocab_size)

    source_pos = PositionalEncoding(model_dimension, source_context_size, dropout)
    target_pos = PositionalEncoding(model_dimension, target_context_size, dropout)

    encoder_blocks = []
    for _ in range(number_of_blocks):
        attn = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        ff   = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
        encoder_blocks.append(EncoderBlock(model_dimension, attn, ff, dropout))

    decoder_blocks = []
    for _ in range(number_of_blocks):
        self_attn  = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        cross_attn = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        ff         = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
        decoder_blocks.append(
            DecoderBlock(model_dimension, self_attn, cross_attn, ff, dropout))

    encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))
    decoder = Decoder(model_dimension, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(model_dimension, N_SQUARES)  # 64 izlazna logita

    transformer = Transformer(
        encoder, decoder,
        source_embed, target_embed,
        source_pos, target_pos,
        projection_layer
    )

    # Xavier inicijalizacija 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameteres: {total:,}")
    print(f"Trainable:       {trainable:,}")
    return trainable


if __name__ == '__main__':
    print("Test GazeTransformer (Encoder-Decoder)")
    print("=" * 45)

    model = build_transformer()
    count_parameters(model)

    B = 4
    # Source: (batch, seq_len) — (0-63, 64=PAD)
    src = torch.randint(0, 64, (B, 20))
    src[:, :5] = PAD_IDX   # prvih 5 = PAD

    tgt = torch.full((B, 1), SOS_IDX, dtype=torch.long)

    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(1).int()
    tgt_mask = causal_mask(tgt.size(1))

    enc_out = model.encode(src, src_mask)
    dec_out = model.decode(enc_out, src_mask, tgt, tgt_mask)
    logits  = model.project(dec_out[:, -1])

    print(f"\nEncoder input: {src.shape}  (batch, seq_len)")
    print(f"Decoder input: {tgt.shape}  (batch, 1 SOS token)")
    print(f"Output logits: {logits.shape}  ([{B}, {N_SQUARES}])")
    assert logits.shape == (B, N_SQUARES)
    print("\nTests passed!")