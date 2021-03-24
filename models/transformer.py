# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import pdb
#pdb.set_trace()
#from attenion_layer import GaussianMultiheadAttention
from torch.nn.functional import linear, pad
from torch.nn import MultiheadAttention
from typing import Optional, Tuple, List
import warnings


def multi_head_attention_forward(
                                 query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 gaussian: Optional[Tensor] = None,
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias

                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling


    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
#    pdb.set_trace()
    naive = True
    if naive:
       attn_output_weights = torch.bmm(q, k.transpose(1, 2))
       assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

       if attn_mask is not None:
         if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
         else:
            attn_output_weights += attn_mask


       if key_padding_mask is not None:
          attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
          attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
          )
          attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
       attn_output_weights = attn_output_weights +  gaussian[0].permute(2, 0, 1)
       attn_output_weights = torch.nn.functional.softmax(
           attn_output_weights, dim=-1)
       attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=dropout_p, training=training)

       attn_output = torch.bmm(attn_output_weights, v)
       assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]


    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output, attn_output_weights.view(bsz, num_heads, tgt_len, src_len)


class GaussianMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(GaussianMultiheadAttention, self).__init__(embed_dim, num_heads, **kwargs)
        self.gaussian = True
#        self.attention = ClusteredAttention(group_Q, group_K, attention_dropout=self.dropout)
#        self.naive = None
        #self.attention = ImprovedClusteredAttention(1000, softmax_temp=1)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, gaussian=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, gaussian=gaussian)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, gaussian=gaussian)




class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, smooth=8, dynamic_scale=True,head_mixture=True, query_mixture=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layers = []
        for layer_index in range(num_decoder_layers):
            decoder_layer = TransformerDecoderLayer(query_mixture, head_mixture, dynamic_scale, smooth, layer_index, d_model, nhead, dim_feedforward,
                                               dropout, activation, normalize_before)
            decoder_layers.append(decoder_layer)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
#        pdb.set_trace()        
#        self.prior = self.input_proj = nn.Conv2d(256, 100, kernel_size=1)  
        if dynamic_scale=="type2" or dynamic_scale=="type3" or dynamic_scale=="type4":
           nn.init.zeros_(self.decoder.layers[0].point3.weight)
           nn.init.zeros_(self.decoder.layers[1].point3.weight)
           nn.init.zeros_(self.decoder.layers[2].point3.weight)
           nn.init.zeros_(self.decoder.layers[3].point3.weight)
           nn.init.zeros_(self.decoder.layers[4].point3.weight)
           nn.init.zeros_(self.decoder.layers[5].point3.weight)
           with torch.no_grad():
                nn.init.ones_(self.decoder.layers[0].point3.bias)
                nn.init.ones_(self.decoder.layers[1].point3.bias)
                nn.init.ones_(self.decoder.layers[2].point3.bias)
                nn.init.ones_(self.decoder.layers[3].point3.bias)
                nn.init.ones_(self.decoder.layers[4].point3.bias)
                nn.init.ones_(self.decoder.layers[5].point3.bias)
#                if dynamic_scale=="type4":
#                   nn.init.constant_(self.decoder.layers[0].point3.bias, 0.5)
#                   nn.init.constant_(self.decoder.layers[1].point3.bias, 0.5)
#                   nn.init.constant_(self.decoder.layers[2].point3.bias, 0.5)
#                   nn.init.constant_(self.decoder.layers[3].point3.bias, 0.5)
#                   nn.init.constant_(self.decoder.layers[4].point3.bias, 0.5)
#                   nn.init.constant_(self.decoder.layers[5].point3.bias, 0.5)
#                   self.decoder.layers[0].point3.bias = self.decoder.layers[0].point3.bias.data * 0.5
#                   self.decoder.layers[1].point3.bias = self.decoder.layers[1].point3.bias.data * 0.5
#                   self.decoder.layers[2].point3.bias = self.decoder.layers[2].point3.bias.data * 0.5
#                   self.decoder.layers[3].point3.bias = self.decoder.layers[3].point3.bias.data * 0.5
#                   self.decoder.layers[4].point3.bias = self.decoder.layers[4].point3.bias.data * 0.5
#                   self.decoder.layers[5].point3.bias = self.decoder.layers[5].point3.bias.data * 0.5
##            self.decoder.layers[0].point2.bias[2:].copy_(torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, 1]))
#            self.decoder.layers[1].point2.bias[2:].copy_(torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, 1]))
#            self.decoder.layers[2].point2.bias[2:].copy_(torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, 1]))
#            self.decoder.layers[3].point2.bias[2:].copy_(torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, 1]))
#            self.decoder.layers[4].point2.bias[2:].copy_(torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, 1]))
#            self.decoder.layers[5].point2.bias[2:].copy_(torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, 1]))
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, h_w):
        # flatten NxCxHxW to HWxNxC
#        pdb.set_trace()
        bs, c, h, w = src.shape
       

#        attn = self.prior(src)
#        prior = torch.matmul((attn.reshape(bs, 100, -1) + mask.reshape(bs, -1).unsqueeze(1) * (-1000000.0)).softmax(-1), src.reshape(bs, c, -1).transpose(2, 1)).transpose(1, 0)
 
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(src)
        grid = grid.unsqueeze(0).permute(0, 3, 1, 2).flatten(2).permute(2, 0, 1)
        grid = grid.repeat(1, bs * 8 , 1)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
#        tgt = tgt + prior
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, points = self.decoder(grid, h_w, tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), points.transpose(1, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,grid,  h_w, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        points = []
        point_sigmoid_ref = None
        for layer in self.layers:
            output, point, point_sigmoid_ref = layer(grid, h_w, output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, point_ref_previous=point_sigmoid_ref)
#            pdb.set_trace()
            points.append(point)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(points)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, query_mixture, head_mixture, dynamic_scale,  smooth, layer_index, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = GaussianMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.smooth = smooth
        self.dynamic_scale = dynamic_scale
        self.query_mixture = query_mixture
        self.head_mixture = head_mixture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
#        self.dropout4 = nn.Dropout(dropout) 
#        self.point1 = nn.Linear(d_model, d_model)
        
        if layer_index == 0:
           self.point1 = MLP(256, 256, 2, 3)
           self.point2 = nn.Linear(d_model, 2 * 8)
        else:
           self.point2 = nn.Linear(d_model, 2 * 8)
        self.layer_index = layer_index
        if self.dynamic_scale=="type2":
           self.point3 = nn.Linear(d_model, 8)
        elif self.dynamic_scale=="type3":
           self.point3 = nn.Linear(d_model, 16)
        elif self.dynamic_scale=="type4":
           self.point3 = nn.Linear(d_model, 24)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, grid, h_w, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     point_ref_previous: Optional[Tensor] = None):
#        pdb.set_trace()
        tgt_len = tgt.shape[0]
        bs = tgt.shape[1]
        out = self.norm4(tgt+query_pos)        
#        out = tgt + query_pos
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
#        pdb.set_trace()
##        point_inter = self.point2(F.relu(self.norm4(self.point1(tgt+query_pos))))
#        out = F.relu(self.norm4(tgt+query_pos))
#        out = F.relu(self.norm4(self.point1(tgt + query_pos)))
#        out = self.norm4(tgt+query_pos)
#        out = self.dropout4(out)
        point_inter = self.point2(out)

#        point_inter = point_inter.view(100, -1, 2)
#        point_inter = self.point2(F.relu(self.norm4(self.point1(tgt))))
#        point_inter = GradMultiply.apply(point_inter, 0.1)
        if self.layer_index == 0:
           point_sigmoid_ref_inter = self.point1(out)
           point_sigmoid_ref = point_sigmoid_ref_inter.sigmoid()
#           point_sigmoid_ref = point_inter[:, :, :2].sigmoid()
           point_sigmoid_ref = (h_w - 0) * point_sigmoid_ref / 32
           point_sigmoid_ref = point_sigmoid_ref.repeat(1, 1, 8)
#           point_sigmoid_offset  = point_inter[:, :, 2:]
           point_sigmoid_offset = point_inter
        else:
           point_sigmoid_ref = point_ref_previous
           point_sigmoid_offset = point_inter
        point = point_sigmoid_ref + point_sigmoid_offset
#######        point1, point2 = point.split(16, -1)
#        point = h_w * point_inter.sigmoid() / 32
#        point = GradMultiply.apply(point, 0.1)
        point = point.view(tgt_len, -1, 2)
#######        point2 = point2.contiguous().view(tgt_len, -1, 2)
#        pdb.set_trace()

        distance = (point.unsqueeze(1) - grid.unsqueeze(0))**2
#        pdb.set_trace()
        if self.dynamic_scale == "type1":
#           print("dynamic_scale")
           scale = 1
           distance = distance.sum(-1) * scale
        elif self.dynamic_scale == "type2":
           scale = self.point3(out)
           scale = scale * scale
########           scale = scale.reshape(tgt_len, -1).unsqueeze(1)
           scale = scale.reshape(tgt_len, -1).unsqueeze(1)
           distance = distance.sum(-1) * scale           
        elif self.dynamic_scale == "type3":
           scale = self.point3(out)
           scale = scale * scale
           scale = scale.reshape(tgt_len, -1, 2).unsqueeze(1)
           distance = (distance * scale).sum(-1)
        elif self.dynamic_scale == "type4":
#           pdb.set_trace()
           scale = self.point3(out)
           scale = scale * scale
           scale = scale.reshape(tgt_len, -1, 3).unsqueeze(1)

           distance = torch.cat([distance, torch.prod(distance, dim=-1, keepdim=True)], dim=-1)
           distance = (distance * scale).sum(-1)
#           distance = torch.matmul(distance.unsqueeze(3), scale).squeeze(3).sum(-1) 
#        scale = GradMultiply.apply(scale, 0.1)
#        pdb.set_trace()
#        grid = grid.repeat(1, 8, 1)
#        gaussian = -torch.sum((point.unsqueeze(1) - grid.unsqueeze(0))**2, dim=-1) * 0.1
#        gaussian = -0.5 * (torch.sum((point.unsqueeze(1) - grid.unsqueeze(0))**2, dim=-1).sqrt() - 1).abs()
#        if self.dynamic_scale == "type1"
    
#        distance = (point.unsqueeze(1) - grid.unsqueeze(0))**2
#        distance = (distance * scale).sum(-1)
######        distance = torch.sum((point.unsqueeze(1) - grid.unsqueeze(0))**2, dim=-1)
#####        distance2 = torch.sum((point2.unsqueeze(1) - grid.unsqueeze(0))**2, dim=-1)
#        distance = distance[:, :, :bs*8] + distance[:, :, bs*8:]
#        distance = 0.5 * distance
#        pdb.set_trace()
        gaussian = -(distance - 0).abs() / self.smooth
#####        gaussian2 = -0.5 * (distance2 / scale.reshape(tgt_len, -1).unsqueeze(1) - 0).abs()
#        gaussian = gaussian * torch.tensor([0,0,0,0,0,0,0,0]).repeat(bs).unsqueeze(0).unsqueeze(0).to(tgt.device).float()
        if self.head_mixture:
#           print("head_mixture")
           gaussian = gaussian * torch.tensor([1,1,1,1,0,0,0,0]).repeat(bs).unsqueeze(0).unsqueeze(0).to(tgt.device).float()
        if self.query_mixture:
#           print("query_mixture")
           gaussian = torch.cat([torch.zeros(tgt_len // 2), torch.ones(tgt_len // 2)]).unsqueeze(-1).unsqueeze(-1).to(gaussian.device) * gaussian
##        gaussian = -0.5 * (torch.sum((point.unsqueeze(1) - grid.unsqueeze(0))**2, dim=-1).sqrt() - 0).abs()
#        variance = 10 
#        gaussian = torch.exp(-torch.sum((point.unsqueeze(1) - grid.unsqueeze(0))**2, dim=-1) / variance) * (1./(2.*math.pi* variance))
#        sum_gaussian = gaussian.sum(dim=1, keepdim=True)
#        gaussian = gaussian / sum_gaussian
#        point[:, :, 0] = height * point[:, :, 0]
#        point[:, :, 1] = width * point[:, :, 1]
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, gaussian=[gaussian])
#        pdb.set_trace()
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.layer_index == 0:
           return tgt, point_sigmoid_ref_inter, point_sigmoid_ref
        else:
           return tgt, point_sigmoid_ref[:, :, :2], point_sigmoid_ref
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, grid, h_w, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                point_ref_previous: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(grid, h_w, tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, point_ref_previous)
        return self.forward_post(grid, h_w, tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, point_ref_previous)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        smooth=args.smooth,
        dynamic_scale=args.dynamic_scale,
        head_mixture=args.head_mixture,
        query_mixture=args.query_mixture,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
