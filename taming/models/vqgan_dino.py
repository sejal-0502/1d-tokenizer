import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from taming.util import instantiate_from_config
from einops.layers.torch import Rearrange
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.diffusionmodules.model_vit import EncoderVIT, DecoderDINOCNN
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
# from taming.modules.vqvae.quantize import VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.models.vqgan import VQModel, VQModel2
from taming.models.vqgan_with_entropy_loss import VQModel2WithEntropyLoss
from taming.modules.vqvae.quantize_with_entropy_loss import VectorQuantizer2WithEntropyLoss
from timm.models.vision_transformer import VisionTransformer

import timm
from einops import rearrange
from torchvision import models
from timm.layers.pos_embed import resample_abs_pos_embed

class ResNet50Block4Features(models.ResNet):
    def __init__(self, pretrained_model):
        super(ResNet50Block4Features, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3])
        #remove fc layer from Resnet50
        self.fc = torch.nn.Identity()
        self.load_state_dict(pretrained_model.state_dict())  

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Block groups
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x) # 8x8x2048

        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return features

class VQModel2WithEntropyDINOLoss(VQModel2WithEntropyLoss):
    def __init__(self, 
                 encoder_config,
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
                 dino_model_type = 'VIT_DINOv2', 
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, 
                         grad_acc_steps, ckpt_path, ignore_keys, image_key, colorize_nlabels, 
                         monitor, 
                         entropy_loss_weight_scheduler_config, )
        n_embed = quantizer_config.params['n_e']
        embed_dim = quantizer_config.params['e_dim']
        image_size = encoder_config.params['image_size']
        quantizer_normalize_embedding = quantizer_config.params.get("normalize_embedding", False)
        self.num_latent_tokens = encoder_config.params['num_latent_tokens']
        self.loss = instantiate_from_config(loss_config)
        self.quantizer_normalize_embedding = quantizer_normalize_embedding
        self.quantize = instantiate_from_config(quantizer_config)
        self.entropy_loss_weight_scheduler = instantiate_from_config(entropy_loss_weight_scheduler_config)
        self.grad_acc_steps = grad_acc_steps
        self.model_width = encoder_config.params['model_width']
        self.scale = self.model_width ** -0.5
        # self.latent_tokens = nn.Parameter(self.scale * torch.randn(self.num_latent_tokens, self.encoder.model_width))

        self.encoder_normalize_embedding = encoder_config.params.get("normalize_embedding", False)

    def entropy_loss_weight_scheduling(self):
        self.loss.entropy_loss_weight = self.entropy_loss_weight_scheduler(self.global_step)

    def encode(self, x):
        h = self.encoder(x)
        if self.encoder_normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):

        return self.decoder(quant)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        # print("----------Quant: ------------", quant.shape)
        dec = self.decode(quant)
        # print("----------Decoder: ------------", dec.shape)
        return dec, diff, _

    def dino_loss(self, dino_output, decoder_dino_output):
        dino_output = dino_output[:, 1:, :]
        dino_output = dino_output.permute(0, 2, 1).contiguous()
        decoder_dino_output = decoder_dino_output.view(decoder_dino_output.shape[0], decoder_dino_output.shape[1], -1)
        cos_similarity = F.cosine_similarity(decoder_dino_output, dino_output, dim=1)
        cosine_loss = 1 - cos_similarity
        dino_loss = cosine_loss.mean()
        return dino_loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  # list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  # list(self.post_quant_conv_dino.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))

        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.loss.min_lr_multiplier)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps, self.loss.min_lr_multiplier)
        
        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]

    def training_step(self, batch, batch_idx):
        self.entropy_loss_weight_scheduling()
        self.log("train/enropy_loss_weight", self.loss.entropy_loss_weight, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        opt_ae, opt_disc = self.optimizers()
        [scheduler_ae_warmup, scheduler_disc_warmup] = self.lr_schedulers()

        x = self.get_input(batch, self.image_key)
        xrec, qloss, decoder_dino_output = self(x)

        optimizer_idx = 1
        discloss, log_dict_disc = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=1, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        discloss = discloss / self.grad_acc_steps
        self.manual_backward(discloss)
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            scheduler_disc_warmup.step()

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=0, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, decoder_dino_output = self(x)

        aeloss, log_dict_ae = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=0, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        discloss, log_dict_disc = self.loss(codebook_entropy_losses=qloss, dino_loss=None, inputs=x, reconstructions=xrec, optimizer_idx=1, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc)
        return self.log_dict

class VisionTransformerWithPretrainedWts(VisionTransformer):
    def __init__(self, patch_size, img_size, num_extra_tokens,  **kwargs):
        """
        pretrained_cfg: pass the same kwargs you’d pass to timm.create_model
        num_extra_tokens: how many extra tokens to prepend
        """
        super().__init__(img_size=img_size,**kwargs)
        self.num_extra_tokens = num_extra_tokens
        if num_extra_tokens > 0:
            self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
            # 1) create the extra tokens
            self.prompt_tokens = nn.Parameter(torch.zeros(1, num_extra_tokens, self.embed_dim))
            # 2) replace the pos_embed with a larger one
            num_patches = (img_size // patch_size) ** 2
            # print("Patch count : ", num_patches)
            new_len = 1 + num_extra_tokens + num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, new_len, self.embed_dim))

    def forward_features(self, x):
        B = x.size(0)
        x = self.patch_embed.proj(x)                    # (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        if self.num_extra_tokens > 0:
            prompt_tokens = self.prompt_tokens.expand(B, -1, -1)  # (B,K,D)
            # x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
            x = torch.cat((cls_tokens, x, prompt_tokens), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
# def resize_patch_embedding_weights(old_weight, new_size):
#     """
#     Resize patch embedding weights using bilinear interpolation.
#     old_weight: [C_out, C_in, H_old, W_old]
#     new_size: (H_new, W_new)
#     """
#     C_out, C_in, H_old, W_old = old_weight.shape
#     H_new, W_new = new_size
#     # Reshape to [C_out * C_in, 1, H, W] for interpolation
#     old_weight_reshaped = old_weight.reshape(-1, 1, H_old, W_old)
#     new_weight = F.interpolate(old_weight_reshaped, size=(H_new, W_new), mode='bilinear', align_corners=True)
#     return new_weight.reshape(C_out, C_in, H_new, W_new)


"""Used class - Changes in the class for removing the pretrained wts"""
class VQModel2WithEntropyDINOLossMAEinit(VQModel2WithEntropyDINOLoss):
    def __init__(self, 
                 pretrained_model_name,
                 encoder_config,
                 decoder_config,
                 quantizer_config,
                 loss_config,
                 grad_acc_steps=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 entropy_loss_weight_scheduler_config=None
                 ):
        super().__init__(encoder_config, decoder_config, quantizer_config, loss_config, 
                         grad_acc_steps, ckpt_path, ignore_keys, image_key, colorize_nlabels, 
                         monitor, 
                         entropy_loss_weight_scheduler_config 
                        #  dino_model_type
                         )
        self.patch_size = encoder_config.params['patch_size']
        self.image_size = encoder_config.params['image_size']
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_latent_tokens = encoder_config.params['num_latent_tokens']
        self.encoder.width = encoder_config.params['model_width']
        self.width = encoder_config.params['model_width']
        self.pretrained = encoder_config.params['pretrained']
        self.scale = scale = self.width ** -0.5
        self.grid_size = self.image_size // self.patch_size
        self.pretrained_wt = 768
        self.token_size = encoder_config.params['token_size']
        self.latent_tokens = nn.Parameter(self.scale * torch.randn(self.num_latent_tokens, self.encoder.width)) 
        self.ln = nn.Linear(self.pretrained_wt, self.token_size)
        # self.latent_tokens = nn.Parameter(self.scale * torch.randn(self.num_latent_tokens, self.encoder.width))

        if pretrained_model_name:
            if pretrained_model_name == 'MAE' :
                pretrained_encoder_model = 'vit_base_patch16_224.mae'
            elif pretrained_model_name == 'DINO_TITOK' :
                pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
            elif pretrained_model_name == 'CLIP' :
                pretrained_encoder_model = 'timm/vit_base_patch16_clip_224.dfn2b'
            elif pretrained_model_name == 'DINO_V2':
                pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
            elif pretrained_model_name == 'DEPTH_ANYTHING_V2' :
                pretrained_encoder_model = '/work/dlclarge2/mutakeks-storage_titok/data/depth_anything/depth_anything_v2_vitb.pth?download=true'

        if self.pretrained == True:
            if pretrained_model_name != 'DEPTH_ANYTHING_V2':
                pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
                state_dict = pretrained_model.state_dict()
                # print('State dict pos embeddings before : ', state_dict['pos_embed'].shape)

                #Create your model instance (with any architectural tweaks already applied)
                self.encoder = VisionTransformerWithPretrainedWts(patch_size=self.patch_size, img_size=self.image_size, num_extra_tokens=self.num_latent_tokens)
                # print("Pos embedding of self.encoder : ", self.encoder.pos_embed.shape)
                K = self.encoder.num_extra_tokens
                # print("Shape of K : ", K)
                state_dict['pos_embed'] = nn.Parameter(torch.zeros(1, 1+K+(state_dict['pos_embed'].shape[1]-1), 768))  # (1, 1+K+N, D)


                if pretrained_model_name == 'CLIP':
                    state_dict.pop("head.weight", None)
                    state_dict.pop("head.bias", None)
                # 3. Load weights, ignoring any missing or unexpected keys
                # print("Implemented until here 1")
                missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
                # print("Implemented until here 2")
                # for name, param in self.encoder.named_parameters():
                #     print(f"{name}: {param.shape}")

                print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
                print("Missing keys:")
                print(missing)
                print("Unexpected Keys: ")
                print(unexpected)
            else :
                pretrained_model = torch.load(pretrained_encoder_model, map_location='cuda')
                if 'state_dict' in pretrained_model:
                    state_dict = pretrained_model['state_dict']
                else:
                    state_dict = pretrained_model

                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("pretrained.", "")  # remove 'module.' if present
                    new_state_dict[new_key] = v
                state_dict = new_state_dict

                # print('State dict pos embeddings before : ', state_dict['pos_embed'].shape)

                pos_embed = state_dict['pos_embed']
                embedding_dim = pos_embed.shape[-1]
                cls_token = pos_embed[:, :1, :]
                patch_pos_embed = pos_embed[:, 1:, :]

                old_size = int(patch_pos_embed.shape[1] ** 0.5)     # 37
                patch_pos_embed = patch_pos_embed.reshape(1, old_size, old_size, embedding_dim).permute(0, 3, 1, 2)

                new_size = (self.image_size // self.patch_size)
                interpolated = F.interpolate(patch_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # Reshape back to [1, 256, D]
                new_patch_pos_embed = interpolated.permute(0, 2, 3, 1).reshape(1, new_size * new_size, embedding_dim)

                # Concatenate cls token back
                new_pos_embed = torch.cat((cls_token, new_patch_pos_embed), dim=1)  # [1, 1+256, D]

                # Replace in state_dict
                state_dict['pos_embed'] = new_pos_embed

                # print('State dict pos embeddings after 1 : ', state_dict['pos_embed'].shape)

                #Create your model instance (with any architectural tweaks already applied)
                self.encoder = VisionTransformerWithPretrainedWts(patch_size=self.patch_size, img_size=self.image_size, num_extra_tokens=self.num_latent_tokens)
                K = self.encoder.num_extra_tokens
                state_dict['pos_embed'] = nn.Parameter(torch.zeros(1, 1+K+(state_dict['pos_embed'].shape[1]-1), 768))  # (1, 1+K+N, D)
                # print('State dict pos embeddings after 2 : ', state_dict['pos_embed'].shape)

                # 3. Load weights, ignoring any missing or unexpected keys
                missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
                # for name, param in self.encoder.named_parameters():
                #     print(f"{name}: {param.shape}")

                print(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
                print("Missing keys:")
                print(missing)
                print("Unexpected Keys: ")
                print(unexpected)
        else :
            print("Not using pretrained wts")
            self.encoder = instantiate_from_config(encoder_config)

    def encode(self, x):
        if self.pretrained == True :
            h = self.encoder.forward_features(x)
            # print("Shape of h : ", h.shape)
            h = h.unsqueeze(2)
            h = h[:, 1+self.grid_size**2:, :] 
            h = self.ln(h)
            h = h.permute(0, 3, 2, 1)
            if self.encoder_normalize_embedding:
                h = F.normalize(h, p=2, dim=1)
        else :
            h = self.encoder(x)
            if self.encoder_normalize_embedding:
                h = F.normalize(h, p=2, dim=1)

        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

class VQModelDino(VQModel):
    def __init__(self,
                 ddconfig,
                 encoder_params,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 encoder_model='CNN',
                 decoder_model='CNN',
                 normalize_embedding=False,
                 dino_dist=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig, lossconfig, n_embed, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor, remap, sane_index_shape)   
        self.automatic_optimization = False
        self.image_key = image_key
        self.normalize_embedding = normalize_embedding
        self.dino_dist = dino_dist
        self.decoder_model = decoder_model
        self.encoder_model = encoder_model
        if self.encoder_model == 'CNN':
            self.encoder = Encoder(**ddconfig)
        elif self.encoder_model == 'VIT':
            self.encoder = EncoderVIT(**encoder_params)

        if self.decoder_model == 'CNN':
            raise NotImplementedError("Decoder with dino distillation not implemented.")
            self.decoder = Decoder(**ddconfig)

        self.dino = timm.create_model('timm/vit_base_patch16_224.dino', pretrained=True).cuda().eval() # DINO VIT-Base emb_dim:768
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.normalize_embedding:
            h = F.normalize(h, p=2, dim=1)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        if self.dino_dist:
            return self.decoder(quant, dino_dist=True)
        else:
            return self.decoder(quant)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        if self.dino_dist:
            dec, dino_conv_out = self.decode(quant)
            return dec, diff, dino_conv_out
        else:
            dec = self.decode(quant)
            return dec, diff

    def dino_loss(self, dino_output, decoder_dino_output):
        dino_output = dino_output[:, 1:, :] # uncomment for DINOv1 
        dino_output = dino_output.permute(0, 2, 1).contiguous()
        decoder_dino_output = decoder_dino_output.view(decoder_dino_output.shape[0], decoder_dino_output.shape[1], -1)
        cos_similarity = F.cosine_similarity(decoder_dino_output, dino_output, dim=2)
        cosine_loss = 1 - cos_similarity
        dino_loss = cosine_loss.mean()
        return dino_loss

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        if self.dino_dist:
            xrec, qloss, decoder_dino_output = self(x)
            with torch.no_grad():
                dino_output = self.dino.forward_features(x)
                
            dino_loss = self.dino_loss(dino_output, decoder_dino_output)
        else:
            xrec, qloss = self(x)
            dino_loss = torch.tensor([0.0])
        opt_ae, opt_disc = self.optimizers()

        optimizer_idx = 1
        # discriminator
        discloss, log_dict_disc = self.loss(qloss, dino_loss, x, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        optimizer_idx = 0
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, dino_loss, x, xrec, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        if self.dino_dist:
            self.log("train/dino_loss", dino_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        if self.dino_dist:
            xrec, qloss, decoder_dino_output = self(x)
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                dino_output = self.dino.forward_features(x)
                dino_loss = self.dino_loss(dino_output, decoder_dino_output)
        else:
            xrec, qloss = self(x)
            dino_loss = torch.tensor([0.0])
        
        aeloss, log_dict_ae = self.loss(qloss, dino_loss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        if self.dino_dist:
            self.log("val/dino_loss", dino_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        discloss, log_dict_disc = self.loss(qloss, dino_loss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if self.dino_dist:
            xrec, _, _ = self(x)
        else:
            xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log