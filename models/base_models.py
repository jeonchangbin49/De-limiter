import torch
import torch.nn as nn
from asteroid.models.base_models import (
    BaseEncoderMaskerDecoder,
    _unsqueeze_to_3d,
    _shape_reconstructed,
)
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape
from einops import rearrange


class BaseEncoderMaskerDecoderWithConfigs(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None, **kwargs):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.use_encoder = kwargs.get("use_encoder", True)
        self.apply_mask = kwargs.get("apply_mask", True)
        self.use_decoder = kwargs.get("use_decoder", True)

    def forward(self, wav):
        """
        Enc/Mask/Dec model forward with some additional options.
        Some of the models we use, like TFC-TDF-UNet, have no masker.
        In UMX or X-UMX, they already use masking in their model implementation.
        Since we do not want to manipulate the model codes, we use this wrapper.

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        if self.use_encoder:
            tf_rep = self.forward_encoder(wav)
        else:
            tf_rep = wav

        est_masks = self.forward_masker(tf_rep)

        if self.apply_mask:
            masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        else:  # model already used masking
            masked_tf_rep = est_masks

        if self.use_decoder:
            decoded = self.forward_decoder(masked_tf_rep)
            reconstructed = pad_x_to_y(decoded, wav)

            return masked_tf_rep, _shape_reconstructed(reconstructed, shape)

        else:  # In UMX or X-UMX, decoder is not used
            decoded = masked_tf_rep

            return decoded


class BaseEncoderMaskerDecoder_mixture_consistency(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(encoder, masker, decoder, encoder_activation)

    def forward(self, wav):
        """Enc/Mask/Dec model forward with mixture consistent output

        References:
        [1] : Wisdom, Scott, et al. "Differentiable consistency constraints for improved deep speech enhancement." ICASSP 2019.
        [2] : Wisdom, Scott, et al. "Unsupervised sound separation using mixture invariant training." NeurIPS 2020.

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = _shape_reconstructed(pad_x_to_y(decoded, wav), shape)

        reconstructed = reconstructed + 1 / reconstructed.shape[1] * (
            wav - reconstructed.sum(dim=1, keepdim=True)
        )

        return reconstructed


class BaseEncoderMaskerDecoderWithConfigsMaskOnOutput(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None, **kwargs):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.use_encoder = kwargs.get("use_encoder", True)
        self.apply_mask = kwargs.get("apply_mask", True)
        self.use_decoder = kwargs.get("use_decoder", True)
        self.nb_channels = kwargs.get("nb_channels", 2)
        self.decoder_activation = kwargs.get("decoder_activation", "sigmoid")
        if self.decoder_activation == "sigmoid":
            self.act_after_dec = nn.Sigmoid()
        elif self.decoder_activation == "relu":
            self.act_after_dec = nn.ReLU()
        elif self.decoder_activation == "relu6":
            self.act_after_dec = nn.ReLU6()
        elif self.decoder_activation == "tanh":
            self.act_after_dec = nn.Tanh()
        elif self.decoder_activation == "none":
            self.act_after_dec = nn.Identity()
        else:
            self.act_after_dec = nn.Sigmoid()

    def forward(self, wav):
        """
        For the De-limit task, we will apply the mask on the output of the decoder.
        We want decoder to learn the sample-wise ratio of the sources.

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)  # (batch, n_channels, time)

        # Real forward
        if self.use_encoder:
            tf_rep = self.forward_encoder(wav)  # (batch, n_channels, freq, time)
        else:
            tf_rep = wav

        if self.nb_channels == 2:
            tf_rep = rearrange(
                tf_rep, "b c f t -> b (c f) t"
            )  # c == 2 when stereo input.
        est_masks = self.forward_masker(tf_rep)  # (batch, 1, freq, time)

        # we are going to apply the mask on the output of the decoder
        if self.use_decoder:
            if self.nb_channels == 2:
                est_masks = rearrange(est_masks, "b 1 f t -> b f t")
            est_masks_decoded = self.forward_decoder(est_masks)
            est_masks_decoded = pad_x_to_y(est_masks_decoded, wav)  # (batch, 1, time)
            est_masks_decoded = self.act_after_dec(
                est_masks_decoded
            )  # (batch, 1, time)
            decoded = wav * est_masks_decoded  # (batch, n_channels, time)

            return (
                est_masks_decoded,
                decoded,
            )

        else:
            decoded = est_masks

            return (decoded,)


class BaseEncoderMaskerDecoderWithConfigsMultiChannelAsteroid(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None, **kwargs):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.use_encoder = kwargs.get("use_encoder", True)
        self.apply_mask = kwargs.get("apply_mask", True)
        self.use_decoder = kwargs.get("use_decoder", True)
        self.nb_channels = kwargs.get("nb_channels", 2)
        self.decoder_activation = kwargs.get("decoder_activation", "none")
        if self.decoder_activation == "sigmoid":
            self.act_after_dec = nn.Sigmoid()
        elif self.decoder_activation == "relu":
            self.act_after_dec = nn.ReLU()
        elif self.decoder_activation == "relu6":
            self.act_after_dec = nn.ReLU6()
        elif self.decoder_activation == "tanh":
            self.act_after_dec = nn.Tanh()
        elif self.decoder_activation == "none":
            self.act_after_dec = nn.Identity()
        else:
            self.act_after_dec = nn.Sigmoid()

    def forward(self, wav):
        """
        Enc/Mask/Dec model forward with some additional options.
        For MultiChannel usage of asteroid-based models. (e.g. ConvTasNet)


        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        if self.use_encoder:
            tf_rep = self.forward_encoder(wav)
        else:
            tf_rep = wav

        if self.nb_channels == 2:
            tf_rep = rearrange(
                tf_rep, "b c f t -> b (c f) t"
            )  # c == 2 when stereo input.
        est_masks = self.forward_masker(tf_rep)

        if self.nb_channels == 2:
            tf_rep = rearrange(tf_rep, "b (c f) t -> b c f t", c=self.nb_channels)

        if self.apply_mask:
            # Since original asteroid implementation of masking includes unnecessary unsqueeze operation, we will do it manually.
            masked_tf_rep = est_masks * tf_rep
        else:
            masked_tf_rep = est_masks

        if self.use_decoder:
            decoded = self.forward_decoder(masked_tf_rep)
            reconstructed = pad_x_to_y(decoded, wav)
            reconstructed = self.act_after_dec(reconstructed)

            return masked_tf_rep, _shape_reconstructed(reconstructed, shape)

        else:
            decoded = masked_tf_rep

            return decoded
