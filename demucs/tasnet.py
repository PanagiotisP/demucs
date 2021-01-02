# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Created on 2018/12
# Author: Kaituo XU
# Modified on 2019/11 by Alexandre Defossez, added support for multiple output channels
# Here is the original license:
# The MIT License (MIT)
#
# Copyright (c) 2018 Kaituo XU
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import capture_init, center_trim

EPS = 1e-8


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes,
                         device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class ConvTasNet(nn.Module):
    @capture_init
    def __init__(self,
                 N=256,
                 L=20,
                 B=256,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 C=2,
                 audio_channels=2,
                 samplerate=22050,
                 norm_type="gLN",
                 causal=False,
                 mask_nonlinear='relu',
                 pad=False,
                 band_num=1,
                 copy_TCN=False,
                 dilation_split=False,
                 cascade=False,
                 skip=False,
                 dwt=False):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
            pad: use of padding in TCN or not
            band_num: how many frequency bands to process. More practically, this is the number of parallel separator/TCN modules
            copy_TCN: copy TCN in each band num TCN or use one with reduced parameters (//2 bottleneck size)
            dilation_split: separate layers within one rack
            cascade: option for dilation split, put layers in sequence than in parallel
            skip: determines TCN output, skip or residual 
        """
        super(ConvTasNet, self).__init__()
        assert N % band_num == 0, 'Encoding dimension must be divisible by band num'
        assert copy_TCN == False or band_num >= 1, 'copy_TCN is available only in multi bands models'
        assert not dilation_split or band_num == 1, 'Each extension must be used on its own'
        assert cascade == False or dilation_split == True, 'cascade is available only in dilation split models'
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.audio_channels = audio_channels
        self.pad = pad
        self.band_num = band_num
        self.copy_TCN = copy_TCN
        self.dilation_split = dilation_split
        self.cascade = cascade
        self.samplerate = samplerate
        self.skip = skip
        self.dwt = dwt
        self.even_pad = None
        # Components

        self.encoder = Encoder(self.L // (1 if not self.dwt else 2), (N if not self.dwt else N // 2), audio_channels)
        if self.dwt:
            self.dwt_layer = DWaveletTransformation()
        if self.dilation_split == True:
            self.separators_num = 2
            assert X % self.separators_num == 0, 'Dilation rates must be split equally between the TCNs'
            self.separators = nn.ModuleList()
            if self.cascade:
                self.separators.append(TemporalConvNet(N, B, H, P, X // self.separators_num, R * self.separators_num, C, norm_type, causal, mask_nonlinear, pad, dilation_group=self.separators_num, skip_non_linearity=True, skip=self.skip, cascade=self.cascade))
            else:
                for i in range(self.separators_num):
                    self.separators.append(TemporalConvNet(N, B, H, P, X // self.separators_num, R, C, norm_type, causal, mask_nonlinear, pad, dilation_group=i, skip_non_linearity=True, skip=self.skip))
        elif self.band_num == 1 and self.dilation_split == False:
            self.separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear, pad, skip=self.skip)
        else:
            self.separators = nn.ModuleList()
            for i in range(band_num):
                if self.copy_TCN:
                    self.separators.append(TemporalConvNet(N // self.band_num, B, H, P, X, R, C, norm_type, causal, mask_nonlinear, pad, skip=self.skip))
                else:
                    self.separators.append(TemporalConvNet(N // self.band_num, B // self.band_num, H, P, X, R, C, norm_type, causal, mask_nonlinear, pad, skip=self.skip))
        self.decoder = Decoder((N if not self.dwt else N // 2), self.L // (1 if not self.dwt else 2), audio_channels)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        if self.pad == False:
            length += self.R * (self.P-1) * (2**self.X - 1) * self.L // 2
        if length % self.L // 2 != 0:
            length += (self.L // 2) - (length % (self.L // 2))
        return length

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        # print('mixture', mixture.shape)
        mixture_w = self.encoder(mixture)
        if self.dwt:
            if mixture_w.shape[-1] % 2 != 0:
                mixture_w = F.pad(mixture_w, (0,1))
                self.even_pad = True
            mixture_w = self.dwt_layer(mixture_w, inverse=False)
        # print('mixture_w', mixture_w.shape)
        split_mixture_w = torch.chunk(mixture_w, self.band_num, dim=1)
        est_masks = []
        est_mask = 0.
        if self.dilation_split == True:
            # Get the two tcn outputs and reduce them either with sum or with product
            # Pass it from non linearity right after.
            # Currently, reducer is manual and non linearity statically set to ReLU
            reducer = 'product'
            # reducer = 'product'
            tcn_output = 0. if reducer == 'sum' else 1.
            for i in range(len(self.separators)):
                if reducer == 'sum':
                    tcn_output += self.separators[i](mixture_w)
                elif reducer == 'product':
                    tcn_output *= F.relu(self.separators[i](mixture_w))
            est_mask = (F.relu(tcn_output) if reducer == 'sum' else tcn_output)
        elif self.band_num == 1 and self.dilation_split == False:
            est_mask = self.separator(mixture_w)
        else:
            for i in range(self.band_num):
                est_masks.append(self.separators[i](split_mixture_w[i]))
            est_mask = torch.cat(est_masks, dim=2)
        # print('est_mask', est_mask.shape)
        mixture_w = torch.unsqueeze(center_trim(mixture_w,est_mask), 1) * est_mask  # [M, C, N, K]
        if self.dwt:
            mixture_w = self.dwt_layer(mixture_w, inverse=True)
            if self.even_pad:
                mixture_w = mixture_w[:,:,:,:-1]
        est_source = self.decoder(mixture_w)
        # print('est_source', est_source.shape)

        # T changed after conv1d in encoder, fix it here
        # T_origin = mixture.size(-1)
        # T_conv = est_source.size(-1)
        # est_source = F.pad(est_source, (0, T_origin - T_conv))
        if self.C==1:
            est_source = torch.cat([mixture.unsqueeze(1) - est_source, est_source], dim=1)
        return est_source


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N, audio_channels):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w

class DWaveletTransformation(nn.Module):
    def __init__(self, p_scale=1, u_scale=0.5, a_scale=np.sqrt(2)):
        super(DWaveletTransformation, self).__init__()
        self.scales = {
            'p': p_scale,
            'u': u_scale,
            'a': a_scale
        }

    def forward(self, z, inverse):
        if not inverse:
            z_odd, z_even = z[:,:,1::2], z[:,:,::2]
            error = z_odd - self.scales['p']*z_even
            signal = z_even + self.scales['u']*error
            error = error / self.scales['a']
            signal = signal * self.scales['a']
            mixture_w = torch.cat((error,signal), 1)
            return mixture_w
        else:
            enc_dim = z.shape[2]//2
            error, signal = z[:, :, :enc_dim, :], z[:, :, enc_dim:, :]
            signal = signal / self.scales['a']
            error = error * self.scales['a']
            z_even = signal - self.scales['u']*error
            z_odd = error + self.scales['p']*z_even

            source_w = torch.zeros((z_even.shape[0], z_even.shape[1], z_even.shape[2], z_even.shape[3]*2)).cuda()

            source_w[:,:,:,1::2] = z_odd
            source_w[:,:,:,::2] = z_even
            return source_w

class Decoder(nn.Module):
    def __init__(self, N, L, audio_channels):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        # Components
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)

    def forward(self, mixture_w):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        # D = W * M
        # source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(mixture_w, 2, 3)  # [M, C, K, N]
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, ac * L]
        m, c, k, _ = est_source.size()
        est_source = est_source.view(m, c, k, self.audio_channels, -1).transpose(2, 3).contiguous()
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x ac x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu', pad=False, dilation_group=0, skip_non_linearity=False, skip=False, cascade=False):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
            pad: Pad or not
            dilation_group: Used in dilation split tests. Helps to define the starting dilation rate
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        self.pad = pad
        self.dilation_group = dilation_group
        self.skip_non_linearity = skip_non_linearity
        self.skip = skip
        self.cascade = cascade
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        dilation_offset = 0
        for r in range(R):
            blocks = []
            if self.dilation_group and r != 0 and r % (R / self.dilation_group) == 0:
                dilation_offset += 1
            for x in range(X):
                if self.cascade:
                    dilation = 2**(x + dilation_offset * X)
                else:
                    dilation = 2**(x + self.dilation_group * X)
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                padding = padding if pad else 0
                blocks += [
                    TemporalBlock(B,
                                  H,
                                  P,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  norm_type=norm_type,
                                  causal=causal,
                                  pad=self.pad,
                                  skip=self.skip)
                ]
            if not self.skip:
                repeats += [nn.Sequential(*blocks)]
            else:
                repeats += [nn.ModuleList(blocks)]
        if not self.skip:
            temporal_conv_net = nn.Sequential(*repeats)
        else:
            temporal_conv_net = nn.ModuleList(repeats)
            
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)
        # Put together
        if not self.skip:
            self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net,
                                     mask_conv1x1)
        else:
            self.network = nn.Sequential(layer_norm, bottleneck_conv1x1)
            self.temporal_conv_net = temporal_conv_net
            self.mask_conv1x1 = mask_conv1x1
    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        if not self.skip:
            score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        else:
            score = self.network(mixture_w)
            skip_connection_sum = 0.
            residual = score
            for rack in self.temporal_conv_net:
                for block in rack:
                    residual, skip = block(residual)
                    skip_connection_sum = skip + (skip_connection_sum if self.pad else center_trim(skip_connection_sum, skip))
            score = self.mask_conv1x1(skip_connection_sum)
        score = score.view(M, self.C, N, -1)  # [M, C*N, K] -> [M, C, N, K]
        if self.skip_non_linearity:
            return score
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 causal=False,
                 pad=False,
                 skip=False):
        super(TemporalBlock, self).__init__()
        self.pad = pad
        self.skip = skip
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding,
                                        dilation, norm_type, causal, pad=self.pad, skip=self.skip)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        if not self.skip:
            out = self.net(x)
            return out + (residual if self.pad else center_trim(residual, out))  # look like w/o F.relu is better than w/ F.relu
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        out, skip = self.net(x)
        return out + (residual if self.pad else center_trim(residual, out)), skip  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 causal=False,
                 pad=False,
                 skip=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        self.pad = pad
        self.skip = skip
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=(padding if self.pad else 0),
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        elif self.skip:
            self.net = nn.Sequential(depthwise_conv, prelu, norm)
            self.pointwise_conv = pointwise_conv
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        if self.skip:
            out = self.net(x)
            residual = self.pointwise_conv(out)
            skip = self.skip_conv(out)
            return residual, skip
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "id":
        return nn.Identity()
    else:  # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


if __name__ == "__main__":
    torch.manual_seed(123)
    M, N, L, T = 2, 3, 4, 12
    K = 2 * T // L - 1
    B, H, P, X, R, C, norm_type, causal = 2, 3, 3, 3, 2, 2, "gLN", False
    mixture = torch.randint(3, (M, T))
    # test Encoder
    encoder = Encoder(L, N)
    encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
    mixture_w = encoder(mixture)
    print('mixture', mixture)
    print('U', encoder.conv1d_U.weight)
    print('mixture_w', mixture_w)
    print('mixture_w size', mixture_w.size())

    # test TemporalConvNet
    separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type=norm_type, causal=causal)
    est_mask = separator(mixture_w)
    print('est_mask', est_mask)

    # test Decoder
    decoder = Decoder(N, L)
    est_mask = torch.randint(2, (B, K, C, N))
    est_source = decoder(mixture_w, est_mask)
    print('est_source', est_source)

    # test Conv-TasNet
    conv_tasnet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type=norm_type)
    est_source = conv_tasnet(mixture)
    print('est_source', est_source)
    print('est_source size', est_source.size())
