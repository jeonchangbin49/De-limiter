import numpy as np
import torch
import torch.nn as nn
from asteroid_filterbanks import make_enc_dec

from asteroid.masknn import TDConvNet

import utils
from .base_models import (
    BaseEncoderMaskerDecoderWithConfigs,
    BaseEncoderMaskerDecoderWithConfigsMaskOnOutput,
    BaseEncoderMaskerDecoderWithConfigsMultiChannelAsteroid,
)


def load_model_with_args(args):
    if args.model_loss_params.architecture == "conv_tasnet_mask_on_output":
        encoder, decoder = make_enc_dec(
            "free",
            n_filters=args.conv_tasnet_params.n_filters,
            kernel_size=args.conv_tasnet_params.kernel_size,
            stride=args.conv_tasnet_params.stride,
            sample_rate=args.sample_rate,
        )
        masker = TDConvNet(
            in_chan=encoder.n_feats_out * args.data_params.nb_channels,  # stereo
            n_src=1,  # for de-limit task.
            out_chan=encoder.n_feats_out,
            n_blocks=args.conv_tasnet_params.n_blocks,
            n_repeats=args.conv_tasnet_params.n_repeats,
            bn_chan=args.conv_tasnet_params.bn_chan,
            hid_chan=args.conv_tasnet_params.hid_chan,
            skip_chan=args.conv_tasnet_params.skip_chan,
            # conv_kernel_size=args.conv_tasnet_params.conv_kernel_size,
            norm_type=args.conv_tasnet_params.norm_type if args.conv_tasnet_params.norm_type else 'gLN',
            mask_act=args.conv_tasnet_params.mask_act,
            # causal=args.conv_tasnet_params.causal,
        )

        model = BaseEncoderMaskerDecoderWithConfigsMaskOnOutput(
            encoder,
            masker,
            decoder,
            encoder_activation=args.conv_tasnet_params.encoder_activation,
            use_encoder=True,
            apply_mask=True,
            use_decoder=True,
            decoder_activation=args.conv_tasnet_params.decoder_activation,
        )
        model.use_encoder_to_target = False

    elif args.model_loss_params.architecture == "conv_tasnet":
        encoder, decoder = make_enc_dec(
            "free",
            n_filters=args.conv_tasnet_params.n_filters,
            kernel_size=args.conv_tasnet_params.kernel_size,
            stride=args.conv_tasnet_params.stride,
            sample_rate=args.sample_rate,
        )
        masker = TDConvNet(
            in_chan=encoder.n_feats_out * args.data_params.nb_channels,  # stereo
            n_src=args.conv_tasnet_params.n_src,  # for de-limit task with the standard conv-tasnet setting.
            out_chan=encoder.n_feats_out,
            n_blocks=args.conv_tasnet_params.n_blocks,
            n_repeats=args.conv_tasnet_params.n_repeats,
            bn_chan=args.conv_tasnet_params.bn_chan,
            hid_chan=args.conv_tasnet_params.hid_chan,
            skip_chan=args.conv_tasnet_params.skip_chan,
            # conv_kernel_size=args.conv_tasnet_params.conv_kernel_size,
            norm_type=args.conv_tasnet_params.norm_type if args.conv_tasnet_params.norm_type else 'gLN',
            mask_act=args.conv_tasnet_params.mask_act,
            # causal=args.conv_tasnet_params.causal,
        )

        model = BaseEncoderMaskerDecoderWithConfigsMultiChannelAsteroid(
            encoder,
            masker,
            decoder,
            encoder_activation=args.conv_tasnet_params.encoder_activation,
            use_encoder=True,
            apply_mask=False if args.conv_tasnet_params.synthesis else True,
            use_decoder=True,
            decoder_activation=args.conv_tasnet_params.decoder_activation,
        )
        model.use_encoder_to_target = False

    return model
