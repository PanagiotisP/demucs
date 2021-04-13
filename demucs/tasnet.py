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


class SeqConvTasNet(nn.Module):
    @capture_init
    def __init__(self,
                 enc_dim=256,
                 transform_window=20,
                 bottle_dim=256,
                 hidden_dim=512,
                 kernel_size=3,
                 stacks=9,
                 racks=3,
                 instr_num=2,
                 audio_channels=2,
                 norm_type="gLN",
                 pad=False,
                 band_num=1,
                 skip=False,
                 dwt=False,
                 dwt_time=True,
                 learnable='none',
                 samplerate=22050):
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
            pad: use of padding in TCN or not
            band_num: Number of frequenyc bands to split processing into
            skip: determines TCN output, skip or residual
            learnable: 'none', 'normal', 'inverse', 'both'
            determines learnable initial transformation layer to be paired with dwt
        """
        super(SeqConvTasNet, self).__init__()

        # Hyper-parameter
        self.instr_num = instr_num
        self.learnable = learnable
        self.dwt_time = dwt_time
        self.samplerate = samplerate
        # Components
        learnable_transform = nn.Sequential(
            nn.Conv1d(audio_channels, audio_channels * 2, kernel_size, padding=1, bias=False),
            nn.AvgPool1d(2, padding=1))
        learnable_inverse_transform = nn.ConvTranspose1d(audio_channels * 2,
                                                         audio_channels,
                                                         kernel_size,
                                                         padding=1,
                                                         stride=2,
                                                         bias=False,
                                                         output_padding=1)
        if self.learnable == 'normal':
            self.learnable_transform = learnable_transform
        elif self.learnable == 'inverse':
            self.learnable_inverse_transform = learnable_inverse_transform
        elif self.learnable == 'both':
            self.learnable_transform = learnable_transform
            self.learnable_inverse_transform = learnable_inverse_transform
        if self.dwt_time:
            self.dwt_layer = DWaveletTransformation()

        self.conv_tasnet1 = ConvTasNet(audio_channels=audio_channels,
                                       stacks=stacks,
                                       pad=pad,
                                       band_num=band_num,
                                       skip=skip,
                                       racks=racks,
                                       hidden_dim=hidden_dim,
                                       bottle_dim=bottle_dim,
                                       enc_dim=enc_dim,
                                       instr_num=instr_num,
                                       transform_window=transform_window,
                                       norm_type=norm_type,
                                       dwt=dwt)
        self.conv_tasnet2 = ConvTasNet(audio_channels=audio_channels,
                                       stacks=stacks,
                                       pad=pad,
                                       band_num=band_num,
                                       skip=skip,
                                       racks=racks,
                                       hidden_dim=hidden_dim,
                                       bottle_dim=bottle_dim,
                                       enc_dim=enc_dim,
                                       instr_num=instr_num,
                                       transform_window=transform_window,
                                       norm_type=norm_type,
                                       dwt=dwt)

    def valid_length(self, length):
        return self.conv_tasnet2.valid_length(length)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        if self.dwt_time:
            # Pass input from transformation
            if self.learnable != 'normal' and self.learnable != 'both':
                dwt_output = self.dwt_layer(mixture, False)
            else:
                dwt_output = self.learnable_transform(mixture)

            # Split channels by half
            low = dwt_output[:, :mixture.shape[1]]
            high = dwt_output[:, mixture.shape[1]:]

            # Process each half individually
            est_source_1 = self.conv_tasnet1(low)
            est_source_2 = self.conv_tasnet2(high)

            # Pass input from inverse transformation
            clean_signals = []
            if self.learnable != 'inverse' and self.learnable != 'both':
                for instr_num in range(self.instr_num):
                    combined_signal = torch.cat(
                        [est_source_1[:, instr_num], est_source_2[:, instr_num]], dim=1)
                    clean_signals.append(self.dwt_layer(combined_signal, True))
            else:
                for instr_num in range(self.instr_num):
                    combined_signal = torch.cat(
                        [est_source_1[:, instr_num], est_source_2[:, instr_num]], dim=1)
                    clean_signals.append(self.learnable_inverse_transform(combined_signal, True))

            est_source = torch.stack(clean_signals, dim=1)
        return est_source


class ConvTasNet(nn.Module):
    @capture_init
    def __init__(self,
                 enc_dim=256,
                 transform_window=20,
                 bottle_dim=256,
                 hidden_dim=512,
                 kernel_size=3,
                 stacks=8,
                 racks=3,
                 instr_num=2,
                 audio_channels=2,
                 norm_type="gLN",
                 pad=False,
                 band_num=1,
                 dilation_split=False,
                 skip=False,
                 dwt=False,
                 samplerate=22050):
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
            audio_channels: Number of audio channels (1 for mono, 2 for stereo)
            norm_type: BN, gLN, cLN
            pad: use of padding in TCN or not
            band_num: how many frequency bands to process. More practically, this is the number
            of parallel separator/TCN modules
            dilation_split: separate layers within one rack
            skip: determines TCN output, skip or residual
        """
        super(ConvTasNet, self).__init__()
        assert enc_dim % band_num == 0, 'Encoding dimension must be divisible by band num'
        assert not dilation_split or band_num == 1, 'Each extension must be used on its own'
        # Hyper-parameter
        self.transform_window = transform_window
        self.kernel_size = kernel_size
        self.stacks = stacks
        self.racks = racks
        self.pad = pad
        self.band_num = band_num
        self.dilation_split = dilation_split
        self.dwt = dwt
        self.even_pad = None
        self.samplerate = samplerate
        # Components
        self.encoder = Encoder(transform_window, enc_dim, audio_channels)
        if dwt:
            self.dwt_layer = DWaveletTransformation()

        self.separators = nn.ModuleList()
        if dilation_split:
            self.separators_num = 2
            assert stacks % self.separators_num == 0, '''Dilation rates must be split
            equally between the TCNs'''
            for i in range(self.separators_num):
                self.separators.append(
                    TemporalConvNet(enc_dim,
                                    bottle_dim,
                                    hidden_dim,
                                    kernel_size,
                                    stacks // self.separators_num,
                                    racks,
                                    instr_num,
                                    norm_type,
                                    pad,
                                    dilation_group=i,
                                    skip=skip))
        elif band_num >= 1:
            for i in range(band_num):
                self.separators.append(
                    TemporalConvNet(enc_dim // band_num,
                                    bottle_dim,
                                    hidden_dim,
                                    kernel_size,
                                    stacks,
                                    racks,
                                    instr_num,
                                    norm_type,
                                    pad,
                                    skip=skip))
        else:
            self.separators.append(
                TemporalConvNet(enc_dim,
                                bottle_dim,
                                hidden_dim,
                                kernel_size,
                                stacks,
                                racks,
                                instr_num,
                                norm_type,
                                pad,
                                skip=skip))

        self.decoder = Decoder(enc_dim, transform_window, audio_channels)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        if not self.pad:
            length += self.racks * (self.kernel_size - 1) * (2**self.stacks -
                                                             1) * self.transform_window // 2
        if length % self.transform_window // 2 != 0:
            length += (self.transform_window // 2) - (length % (self.transform_window // 2))
        return length

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        if self.dwt:
            if mixture_w.shape[-1] % 2 != 0:
                mixture_w = F.pad(mixture_w, (0, 1))
                self.even_pad = True
            mixture_w = self.dwt_layer(mixture_w, inverse=False)
        split_mixture_w = torch.chunk(mixture_w, self.band_num, dim=1)
        est_masks = []
        est_mask = 0.
        if self.dilation_split:
            tcn_output = 0.
            for i in range(len(self.separators)):
                tcn_output += self.separators[i](mixture_w)
            est_mask = F.relu(tcn_output)
        elif self.band_num > 1:
            for i in range(self.band_num):
                est_masks.append(self.separators[i](split_mixture_w[i]))
            est_mask = torch.cat(est_masks, dim=2)
        else:
            # Basic model, only one separator
            est_mask = self.separators[0](mixture_w)
        mixture_w = torch.unsqueeze(center_trim(mixture_w, est_mask), 1) * est_mask  # [M, C, N, K]
        if self.dwt:
            mixture_w = self.dwt_layer(mixture_w, inverse=True)
            if self.even_pad:
                mixture_w = mixture_w[:, :, :, :-1]
        est_source = self.decoder(mixture_w)
        return est_source


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, kernel_size, enc_dim, audio_channels):
        super(Encoder, self).__init__()
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(audio_channels,
                                  enc_dim,
                                  kernel_size=kernel_size,
                                  stride=kernel_size // 2,
                                  bias=False)

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
        self.scales = {'p': p_scale, 'u': u_scale, 'a': a_scale}

    def forward(self, z, inverse):
        if not inverse:
            z_odd, z_even = z[:, :, 1::2], z[:, :, ::2]
            error = z_odd - self.scales['p'] * z_even
            signal = z_even + self.scales['u'] * error
            error = error / self.scales['a']
            signal = signal * self.scales['a']
            mixture_w = torch.cat((error, signal), 1)
            return mixture_w
        else:
            enc_dim = z.shape[2] // 2
            error, signal = z[:, :, :enc_dim, :], z[:, :, enc_dim:, :]
            signal = signal / self.scales['a']
            error = error * self.scales['a']
            z_even = signal - self.scales['u'] * error
            z_odd = error + self.scales['p'] * z_even

            source_w = torch.zeros(
                (z_even.shape[0], z_even.shape[1], z_even.shape[2], z_even.shape[3] * 2)).cuda()

            source_w[:, :, :, 1::2] = z_odd
            source_w[:, :, :, ::2] = z_even
            return source_w


class Decoder(nn.Module):
    def __init__(self, enc_dim, transform_window, audio_channels):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.transform_window = transform_window
        self.audio_channels = audio_channels
        # Components
        self.basis_signals = nn.Linear(enc_dim, audio_channels * transform_window, bias=False)

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
        est_source = overlap_and_add(est_source, self.transform_window // 2)  # M x C x ac x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self,
                 enc_dim,
                 bottle_dim,
                 hidden_dim,
                 kernel_size,
                 stacks,
                 racks,
                 instr_num,
                 norm_type="gLN",
                 pad=False,
                 dilation_group=0,
                 skip=False):
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
            pad: Pad or not
            dilation_group: Used in dilation split tests. Helps to define the starting dilation rate
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.instr_num = instr_num
        self.pad = pad
        self.skip = skip
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(enc_dim)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(enc_dim, bottle_dim, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for _ in range(racks):
            blocks = []
            for x in range(stacks):
                dilation = 2**(x + dilation_group * stacks)
                padding = (kernel_size - 1) * dilation // 2
                padding = padding if pad else 0
                blocks += [
                    TemporalBlock(bottle_dim,
                                  hidden_dim,
                                  kernel_size,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  norm_type=norm_type,
                                  pad=pad,
                                  skip=skip)
                ]
            repeats += [nn.ModuleList(blocks)]
        temporal_conv_net = nn.ModuleList(repeats)

        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(bottle_dim, instr_num * enc_dim, 1, bias=False)
        # Put together
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
        M, N, _ = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        skip_connection_sum = 0.
        residual = score
        for rack in self.temporal_conv_net:
            for block in rack:
                residual, skip = block(residual)
                if self.skip:
                    skip_connection_sum = skip + \
                        (skip_connection_sum if self.pad else center_trim(skip_connection_sum, skip))
        if self.skip:
            score = self.mask_conv1x1(skip_connection_sum)
        else:
            score = self.mask_conv1x1(residual)
        score = score.view(M, self.instr_num, N, -1)  # [M, C*N, K] -> [M, C, N, K]
        est_mask = F.relu(score)
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(self,
                 bottle_dim,
                 hidden_dim,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 pad=False,
                 skip=False):
        super(TemporalBlock, self).__init__()
        self.pad = pad
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(bottle_dim, hidden_dim, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, hidden_dim)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(hidden_dim,
                                        bottle_dim,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        norm_type,
                                        pad=pad,
                                        skip=skip)
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
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        out, skip = self.net(x)
        # look like w/o F.relu is better than w/ F.relu
        return out + (residual if self.pad else center_trim(residual, out)), skip


class DepthwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 pad=False,
                 skip=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        self.skip = skip
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=(padding if pad else 0),
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        # Put together
        self.net = nn.Sequential(depthwise_conv, prelu, norm)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        if skip:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        out = self.net(x)
        residual = self.res_conv(out)

        skip = 0
        if self.skip:
            skip = self.skip_conv(out)
        return residual, skip


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
    print('I do nothing')
