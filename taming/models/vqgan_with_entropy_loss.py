from taming.models.vqgan import VQModel2
from taming.modules.vqvae.quantize_with_entropy_loss import VectorQuantizer2WithEntropyLoss
from taming.util import instantiate_from_config

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQModel2WithEntropyLoss(VQModel2):
    def __init__(self, encoder_config,
                 decoder_config,
                 quantizer_config,
                 loss_config,
                 grad_acc_steps=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 entropy_loss_weight_scheduler_config=None,
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, 
                        loss_config, grad_acc_steps, ckpt_path, ignore_keys, 
                        image_key, colorize_nlabels, monitor)

        self.quantize = instantiate_from_config(quantizer_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)
        self.grad_acc_steps = grad_acc_steps

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss", self.loss.entropy_loss_weight, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return super().training_step(batch, batch_idx)
    
class VQModel2WithEntropyLossMAEinit(VQModel2WithEntropyLoss):
    def __init__(self, encoder_config,
                 decoder_config,
                 quantizer_config,
                 loss_config,
                 grad_acc_steps=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 entropy_loss_weight_scheduler_config=None,
                 pretrained_encoder = 'MAE', 
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, 
                        grad_acc_steps, ckpt_path, ignore_keys, image_key, colorize_nlabels, 
                        monitor, 
                        entropy_loss_weight_scheduler_config, 
                        )
        
        image_size = encoder_config.params['image_size']
        self.patch_size = encoder_config.params['patch_size']
        self.image_size = image_size

        if pretrained_encoder:
            if pretrained_encoder == 'VIT_DINO':
                pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
            elif pretrained_encoder == 'VIT_DINOv2':
                pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
            elif pretrained_encoder == 'MAE':
                pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
            
            self.encoder = timm.create_model(pretrained_encoder_model, img_size=image_size, pretrained=False).train()
            pretrained_model = timm.create_model(pretrained_encoder_model, img_size=image_size, pretrained=True)
            self.initialize_weights(self.encoder, pretrained_model)
        
    def initialize_weights(self, target_model, source_model):
        """Initialize weights of target_model with weights from source_model."""
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data.copy_(source_param.data)

    def encode(self, x):
        h = self.encoder.forward_features(x)[:,1:].permute(0, 2, 1).contiguous()
        num_patches = self.image_size // self.patch_size
        h = h.reshape(h.shape[0], -1, h.shape[2]//num_patches, h.shape[2]//num_patches)
        h = self.quant_conv(h)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info   

class VQModelWithEntopyLoss(VQModel2):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim,
                 ckpt_path=None, ignore_keys=..., image_key="image", colorize_nlabels=None, monitor=None, remap=None, sane_index_shape=False, 
                 entropy_loss_weight_scheduler_config=None,
                 beta=0.25, diversity_gamma = 1., frac_per_sample_entropy=1.,
                 ):
        raise NotImplementedError("This class semms to be deprecated. Use VQModel2WithEntropyLoss instead. Remove if not needed.")
        super().__init__(ddconfig, lossconfig, n_embed, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor, remap, sane_index_shape)
        self.quantize = VectorQuantizer2WithEntropyLoss(n_embed, embed_dim, beta, diversity_gamma=diversity_gamma, frac_per_sample_entropy=frac_per_sample_entropy)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return super().training_step(batch, batch_idx)
