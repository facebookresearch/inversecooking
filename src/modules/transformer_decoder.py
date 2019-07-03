# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Code adapted from https://github.com/pytorch/fairseq
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# https://github.com/pytorch/fairseq. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
import modules.utils as utils
from modules.multihead_attention import MultiheadAttention
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    # creates tensor from scratch - to avoid multigpu issues
    max_pos = padding_idx + 1 + tensor.size(1)
    #if not hasattr(make_positions, 'range_buf'):
    range_buf = tensor.new()
    #make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=range_buf)
    mask = tensor.ne(padding_idx)
    positions = range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)

    out = tensor.clone()
    out = out.masked_scatter_(mask,positions[mask])
    return out


class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step

            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:

            positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        # recompute/expand embeddings if needed
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return self.weights[self.padding_idx + seq_len, :].expand(bsz, 1, -1)

        positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, embed_dim, n_att, dropout=0.5, normalize_before=True, last_ln=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.relu_dropout = dropout
        self.normalize_before = normalize_before
        num_layer_norm = 3

        # self-attention on generated recipe
        self.self_attn = MultiheadAttention(
            self.embed_dim, n_att,
            dropout=dropout,
        )

        self.cond_att = MultiheadAttention(
            self.embed_dim, n_att,
            dropout=dropout,
        )

        self.fc1 = Linear(self.embed_dim, self.embed_dim)
        self.fc2 = Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(num_layer_norm)])
        self.use_last_ln = last_ln
        if self.use_last_ln:
            self.last_ln = LayerNorm(self.embed_dim)

    def forward(self, x, ingr_features, ingr_mask, incremental_state, img_features):

        # self attention
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)

        # attention
        if ingr_features is None:

            x, _ = self.cond_att(query=x,
                                    key=img_features,
                                    value=img_features,
                                    key_padding_mask=None,
                                    incremental_state=incremental_state,
                                    static_kv=True,
                                    )
        elif img_features is None:
            x, _ = self.cond_att(query=x,
                                    key=ingr_features,
                                    value=ingr_features,
                                    key_padding_mask=ingr_mask,
                                    incremental_state=incremental_state,
                                    static_kv=True,
                                    )


        else:
            # attention on concatenation of encoder_out and encoder_aux, query self attn (x)
            kv = torch.cat((img_features, ingr_features), 0)
            mask = torch.cat((torch.zeros(img_features.shape[1], img_features.shape[0], dtype=torch.uint8).to(device),
                              ingr_mask), 1)
            x, _ = self.cond_att(query=x,
                                    key=kv,
                                    value=kv,
                                    key_padding_mask=mask,
                                    incremental_state=incremental_state,
                                    static_kv=True,
            )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(-1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(-1, x, after=True)

        if self.use_last_ln:
            x = self.last_ln(x)

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class DecoderTransformer(nn.Module):
    """Transformer decoder."""

    def __init__(self, embed_size, vocab_size, dropout=0.5, seq_length=20, num_instrs=15,
                 attention_nheads=16, pos_embeddings=True, num_layers=8, learned=True, normalize_before=True,
                 normalize_inputs=False, last_ln=False, scale_embed_grad=False):
        super(DecoderTransformer, self).__init__()
        self.dropout = dropout
        self.seq_length = seq_length * num_instrs
        self.embed_tokens = nn.Embedding(vocab_size, embed_size, padding_idx=vocab_size-1,
                                         scale_grad_by_freq=scale_embed_grad)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_size ** -0.5)
        if pos_embeddings:
            self.embed_positions = PositionalEmbedding(1024, embed_size, 0, left_pad=False, learned=learned)
        else:
            self.embed_positions = None
        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            self.layer_norms_in = nn.ModuleList([LayerNorm(embed_size) for i in range(3)])

        self.embed_scale = math.sqrt(embed_size)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_size, attention_nheads, dropout=dropout, normalize_before=normalize_before,
                                    last_ln=last_ln)
            for i in range(num_layers)
        ])

        self.linear = Linear(embed_size, vocab_size-1)

    def forward(self, ingr_features, ingr_mask, captions, img_features, incremental_state=None):

        if ingr_features is not None:
            ingr_features = ingr_features.permute(0, 2, 1)
            ingr_features = ingr_features.transpose(0, 1)
            if self.normalize_inputs:
                self.layer_norms_in[0](ingr_features)

        if img_features is not None:
            img_features = img_features.permute(0, 2, 1)
            img_features = img_features.transpose(0, 1)
            if self.normalize_inputs:
                self.layer_norms_in[1](img_features)

        if ingr_mask is not None:
            ingr_mask = (1-ingr_mask.squeeze(1)).byte()

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(captions, incremental_state=incremental_state)
        if incremental_state is not None:
            if self.embed_positions is not None:
                positions = positions[:, -1:]
            captions = captions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(captions)

        if self.embed_positions is not None:
            x += positions

        if self.normalize_inputs:
            x = self.layer_norms_in[2](x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for p, layer in enumerate(self.layers):
            x  = layer(
                x,
                ingr_features,
                ingr_mask,
                incremental_state,
                img_features
            )
            
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.linear(x)
        _, predicted = x.max(dim=-1)

        return x, predicted

    def sample(self, ingr_features, ingr_mask, greedy=True, temperature=1.0, beam=-1,
               img_features=None, first_token_value=0,
               replacement=True, last_token_value=0):

        incremental_state = {}

        # create dummy previous word
        if ingr_features is not None:
            fs = ingr_features.size(0)
        else:
            fs = img_features.size(0)

        if beam != -1:
            if fs == 1:
                return self.sample_beam(ingr_features, ingr_mask, beam, img_features, first_token_value,
                                        replacement, last_token_value)
            else:
                print ("Beam Search can only be used with batch size of 1. Running greedy or temperature sampling...")

        first_word = torch.ones(fs)*first_token_value

        first_word = first_word.to(device).long()
        sampled_ids = [first_word]
        logits = []

        for i in range(self.seq_length):
            # forward
            outputs, _ = self.forward(ingr_features, ingr_mask, torch.stack(sampled_ids, 1),
                                      img_features, incremental_state)
            outputs = outputs.squeeze(1)
            if not replacement:
                # predicted mask
                if i == 0:
                    predicted_mask = torch.zeros(outputs.shape).float().to(device)
                else:
                    # ensure no repetitions in sampling if replacement==False
                    batch_ind = [j for j in range(fs) if sampled_ids[i][j] != 0]
                    sampled_ids_new = sampled_ids[i][batch_ind]
                    predicted_mask[batch_ind, sampled_ids_new] = float('-inf')

                # mask previously selected ids
                outputs += predicted_mask

            logits.append(outputs)
            if greedy:
                outputs_prob = torch.nn.functional.softmax(outputs, dim=-1)
                _, predicted = outputs_prob.max(1)
                predicted = predicted.detach()
            else:
                k = 10
                outputs_prob = torch.div(outputs.squeeze(1), temperature)
                outputs_prob = torch.nn.functional.softmax(outputs_prob, dim=-1).data

                # top k random sampling
                prob_prev_topk, indices = torch.topk(outputs_prob, k=k, dim=1)
                predicted = torch.multinomial(prob_prev_topk, 1).view(-1)
                predicted = torch.index_select(indices, dim=1, index=predicted)[:, 0].detach()

            sampled_ids.append(predicted)

        sampled_ids = torch.stack(sampled_ids[1:], 1)
        logits = torch.stack(logits, 1)

        return sampled_ids, logits

    def sample_beam(self, ingr_features, ingr_mask, beam=3, img_features=None, first_token_value=0,
                   replacement=True, last_token_value=0):
        k = beam
        alpha = 0.0
        # create dummy previous word
        if ingr_features is not None:
            fs = ingr_features.size(0)
        else:
            fs = img_features.size(0)
        first_word = torch.ones(fs)*first_token_value

        first_word = first_word.to(device).long()

        sequences = [[[first_word], 0, {}, False, 1]]
        finished = []

        for i in range(self.seq_length):
            # forward
            all_candidates = []
            for rem in range(len(sequences)):
                incremental = sequences[rem][2]
                outputs, _ = self.forward(ingr_features, ingr_mask, torch.stack(sequences[rem][0], 1),
                                          img_features, incremental)
                outputs = outputs.squeeze(1)
                if not replacement:
                    # predicted mask
                    if i == 0:
                        predicted_mask = torch.zeros(outputs.shape).float().to(device)
                    else:
                        # ensure no repetitions in sampling if replacement==False
                        batch_ind = [j for j in range(fs) if sequences[rem][0][i][j] != 0]
                        sampled_ids_new = sequences[rem][0][i][batch_ind]
                        predicted_mask[batch_ind, sampled_ids_new] = float('-inf')

                    # mask previously selected ids
                    outputs += predicted_mask

                outputs_prob = torch.nn.functional.log_softmax(outputs, dim=-1)
                probs, indices = torch.topk(outputs_prob, beam)
                # tokens is [batch x beam ] and every element is a list
                # score is [ batch x beam ] and every element is a scalar
                # incremental is [batch x beam ] and every element is a dict


                for bid in range(beam):
                    tokens = sequences[rem][0] + [indices[:, bid]]
                    score = sequences[rem][1] + probs[:, bid].squeeze().item()
                    if indices[:,bid].item() == last_token_value:
                        finished.append([tokens, score, None, True, sequences[rem][-1] + 1])
                    else:
                        all_candidates.append([tokens, score, incremental, False, sequences[rem][-1] + 1])

            # if all the top-k scoring beams have finished, we can return them
            ordered_all = sorted(all_candidates + finished, key=lambda tup: tup[1]/(np.power(tup[-1],alpha)),
                                 reverse=True)[:k]
            if all(el[-1] == True for el in ordered_all):
                all_candidates = []

            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1]/(np.power(tup[-1],alpha)), reverse=True)
            # select k best
            sequences = ordered[:k]
            finished = sorted(finished,  key=lambda tup: tup[1]/(np.power(tup[-1],alpha)), reverse=True)[:k]

        if len(finished) != 0:
            sampled_ids = torch.stack(finished[0][0][1:], 1)
            logits = finished[0][1]
        else:
            sampled_ids = torch.stack(sequences[0][0][1:], 1)
            logits = sequences[0][1]
        return sampled_ids, logits

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict



def Embedding(num_embeddings, embedding_dim, padding_idx, ):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m
