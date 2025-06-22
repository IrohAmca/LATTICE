from .layer import PositionalEncoding, Embeddings, PositionWiseFeedForward
from .attention import MultiHeadedAttention

from torch import nn


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, tgt, src_mask=None, tgt_mask=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class EncoderTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderTransformerBlock, self).__init__()
        self.attn = MultiHeadedAttention(h=num_heads, d_model=d_model, dropout=dropout)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.attn(x, x, x, mask)

        x = x + self.dropout(attn)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class DecoderTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderTransformerBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(
            h=num_heads, d_model=d_model, dropout=dropout
        )
        self.src_attn = MultiHeadedAttention(
            h=num_heads, d_model=d_model, dropout=dropout
        )
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn)
        x = self.norm1(x)

        src_attn = self.src_attn(x, memory, memory, src_mask)
        x = x + self.dropout(src_attn)
        x = self.norm2(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        e_N,
        d_N,
        d_model,
        d_ff,
        e_num_heads,
        d_num_heads,
        dropout=0.1,
        max_len=5000,
    ):
        super(Transformer, self).__init__()
        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab),
            PositionalEncoding(d_model, dropout, max_len),
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(d_model, tgt_vocab),
            PositionalEncoding(d_model, dropout, max_len),
        )

        self.encoder = nn.ModuleList(
            [
                EncoderTransformerBlock(d_model, e_num_heads, d_ff, dropout)
                for _ in range(e_N)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderTransformerBlock(d_model, d_num_heads, d_ff, dropout)
                for _ in range(d_N)
            ]
        )

        self.generator = Generator(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.src_embed(src)
        for layer in self.encoder:
            memory = layer(memory, src_mask)

        output = self.tgt_embed(tgt)
        for layer in self.decoder:
            output = layer(output, memory, src_mask, tgt_mask)
            
        return self.generator(output)
