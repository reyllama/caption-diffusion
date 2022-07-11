import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils.metrics_accumulator import MetricsAccumulator
from utils.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision import models
import torch.nn.functional as FF
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np

from CLIP import clip
#from classifier import classifier_models
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils.visualization import show_tensor_image, show_editied_masked_image
from BLIP.models.blip import blip_decoder, blip_feature_extractor
import json
from datetime import datetime
# TODO: move BLIP repo under the root dir

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]  # model will contain the first 29 layers

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)

        return features

    def forward_feats(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)

        return features

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        print(args)
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.args.output_path = os.path.join(self.args.output_path, now)
        os.makedirs(self.args.output_path, exist_ok=True)

        with open(os.path.join(self.args.output_path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        if self.args.finetuned:
            blip_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth"
        else:
            blip_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"

        if self.args.guidance == 'clip':
            self.guide_model = (
                clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
            )
            self.guidance_size = self.guide_model.visual.input_resolution
            self.mask_model = blip_decoder(pretrained=blip_path, image_size=384, vit='base').to(self.device)
            self.encoder_model = self.guide_model

        elif self.args.guidance == 'blip':
            self.guide_model = blip_decoder(pretrained=blip_path, image_size=384, vit='base').to(self.device)
            self.guidance_size = self.guide_model.image_size
            self.mask_model = self.guide_model
            self.encoder_model = blip_feature_extractor(pretrained=blip_path, image_size=384, vit='base') # TODO: check if image size 384 is compatible
        else:
            raise ValueError

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)
        if args.vit:
            print("BLIP ViT for style extraction")
            self.style_model = self.mask_model.visual_encoder.eval()
        else:
            print("VGG for style extraction")
            self.style_model = VGG().to(self.device).eval() # Added for style loss computation

        for p in self.mask_model.parameters():
            p.requires_grad = False

        self.image_augmentations = ImageAugmentations(self.guidance_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        self.init_image.requires_grad = False
        self.blur = args.blur

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def blip_loss(self, x_in, text):
        loss = self.guide_model(x_in, text)
        return loss

    def clip_loss(self, x_in, text_embed):
        # incorporate directional loss
        clip_loss = torch.tensor(0)
        x_base = None

        if self.mask is not None:
            self.mask.requires_grad = False
            masked_input = x_in * self.mask
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2) # scaling (-1~1 -> 0~1)
        clip_in = self.normalize(augmented_input)
        image_embeds = self.guide_model.encode_image(clip_in).float()

        if self.args.pseudo_cap:
            x_base = self.init_image
            x_base = x_base.repeat(x_in.size(0), 1, 1, 1)

        if x_base is not None:
            if self.mask is not None:
                masked_base = (x_base * self.mask).detach()
            else:
                masked_base = x_base.detach()
            augmented_base = self.image_augmentations(masked_base).add(1).div(2)
            clip_base = self.normalize(augmented_base)
            base_embeds = self.guide_model.encode_image(clip_base).float()
            dists = d_clip_loss(image_embeds-base_embeds, text_embed)
        else:
            dists = d_clip_loss(image_embeds, text_embed)

        # We want to sum over the averages
        for i in range(self.args.batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

    def unaugmented_clip_distance(self, x, text_embed):
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.guide_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed)

        return dists.item()

    def calc_style_loss(self, gen, style):
        if self.args.vit:
            batch_size, n_patch, n_dim = gen.shape
            style = style.repeat(batch_size, 1, 1)
            G = torch.bmm(gen.permute(0,2,1), gen) # Naive correspondence to ConvNet-based feature correlation
            A = torch.bmm(style.permute(0,2,1), style)
        else:
            batch_size, channel, height, width = gen.shape
            style = style.repeat(batch_size, 1, 1, 1)
            G = torch.bmm(gen.view(batch_size, channel, height * width), gen.view(batch_size, channel, height * width).permute(0,2,1))
            A = torch.bmm(style.view(batch_size, channel, height * width), style.view(batch_size, channel, height * width).permute(0,2,1))

        style_l = torch.norm(G.view(batch_size, -1) - A.view(batch_size, -1), dim=1).mean()
        return style_l

    def style_loss(self, x0, x):
        # print(f"x0: {x0.size()}")
        # print(f"x: {x.size()}")
        if self.args.vit:
            x0 = TF.resize(x0, [self.mask_model.image_size, self.mask_model.image_size])
            x = TF.resize(x, [self.mask_model.image_size, self.mask_model.image_size])
        x0_features = self.style_model.forward_feats(x0)
        x_features = self.style_model.forward_feats(x)
        loss = 0
        for style, gen in zip(x0_features, x_features):
            # extracting the dimensions from the generated image
            # print("style: ", style.size())
            # print("gen: ", gen.size())
            if self.args.ot:
                loss += torch.norm(style-gen)
            else:
                loss += self.calc_style_loss(gen, style)

        return loss

    def pseudo_caption(self):
        image = F.resize(self.init_image, [self.mask_model.image_size, self.mask_model.image_size])
        with torch.no_grad():
            pseudo_cap = self.mask_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) # in text format (un-tokenized)
        return pseudo_cap

    def edit_image_by_prompt(self):

        if self.args.mask_auto:
            mask_np = self.propose_mask()
            mask_binarized = ((mask_np > self.args.mask_thresh) * 255).astype(np.uint8)  # 0.35: woman and her dog / 0.3: t shirts

            if self.blur:
                from scipy.ndimage import filters
                mask_binarized = filters.gaussian_filter(mask_binarized, 0.02 * max(mask_binarized.shape))
                mask_binarized = self.normalize(mask_binarized)

            self.mask = TF.to_tensor(Image.fromarray(mask_binarized))
            self.mask_pil = TF.to_pil_image(self.mask.squeeze().cpu().numpy() * 255)
            self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)
            # self.mask = torch.from_numpy(self.propose_mask()).to(self.device).detach()
            assert self.mask.requires_grad == False, "mask is not updated"
            # self.mask_pil = TF.to_pil_image(self.mask.squeeze().cpu().numpy() * 255)

        elif self.args.mask:
            self.mask_pil = Image.open(self.args.mask).convert("RGB")
            if self.mask_pil.size != self.image_size:
                self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
            image_mask_pil_binarized = ((np.array(self.mask_pil) > 0.5) * 255).astype(np.uint8)
            if self.args.invert_mask:
                image_mask_pil_binarized = 255 - image_mask_pil_binarized
                self.mask_pil = TF.to_pil_image(image_mask_pil_binarized)
            self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
            self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)

        else:
            self.mask = torch.ones_like(self.init_image, device=self.device)
            self.mask_pil = TF.to_pil_image(self.mask.squeeze().cpu().numpy() * 255)

        if self.args.export_assets:
            mask_path = self.assets_path / Path(
                self.args.output_file.replace(".png", "_mask.png")
            )
            self.mask_pil.save(mask_path)

        if self.args.prompt:
            if self.args.guidance == "clip":
                text_embed = self.encoder_model.encode_text(
                    clip.tokenize(self.args.prompt).to(self.device)
                ).float()
            elif self.args.guidance == "blip":
                text_embed = self.encoder_model.text_encoder(
                    self.encoder_model.tokenizer(self.args.prompt, return_tensors="pt").to(self.device).input_ids,
                    attention_mask=None,
                    return_dict=True,
                    mode='text'
                ).last_hidden_state.float()
            else:
                raise ValueError

            ################################################################
            if self.args.pseudo_cap:
                if self.args.mask_base_cap:
                    print(f"Using provided caption: {self.args.mask_base_cap}")
                    pseudo_cap = self.args.mask_base_cap
                else:
                    pseudo_cap = self.pseudo_caption()
                    print(f"Using pseudo caption: {pseudo_cap}")
                pseudo_cap_embed = self.guide_model.encode_text(
                    clip.tokenize(pseudo_cap).to(self.device)
                ).float()

                text_embed = text_embed - pseudo_cap_embed
            ################################################################
        else:
            text_embed = None

        if self.args.export_assets:
            img_path = self.assets_path / Path(self.args.output_file)
            self.init_image_pil.save(img_path)

        def cond_fn(x, t, y=None):
            # TODO: uncommented for now for attribute conditioned synthesis
            # if self.args.prompt == "":
            #     return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                # x_in = out["pred_xstart"]

                loss = torch.tensor(0)
                if text_embed is not None and self.args.guidance=='clip' and self.args.guidance_lambda != 0:
                    clip_loss = self.clip_loss(x_in, text_embed) * self.args.guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())

                elif text_embed is not None and self.args.guidance=='blip' and self.args.guidance_lambda != 0:
                    blip_loss = self.blip_loss(x_in, self.args.prompt) * self.args.guidance_lambda
                    loss = loss + blip_loss
                    self.metrics_accumulator.update_metric("blip_loss", blip_loss.item())

                if self.args.attribute and self.args.attribute_guidance_lambda != 0:
                    attr_loss, attn = self.attribute_loss(x_in)
                    # scale_factor = (x_in.size(-1) / attn.size(-1))**2
                    scale_factor = 1.0
                    attn = FF.interpolate(attn, x_in.shape[-2:]) / scale_factor # (n_attr, batch, H, W)
                    effective_idxs = []
                    for i, v in enumerate(self.args.attribute.split()):
                        if int(v) != -1:
                            effective_idxs += [int(i)]
                    effective_attn = attn[effective_idxs]
                    effective_attn = effective_attn.mean(dim=0).unsqueeze(0) # (1, batch, H, W)
                    effective_attn = effective_attn.mean(dim=1).unsqueeze(1)
                    attr_loss = attr_loss * self.args.attribute_guidance_lambda
                    loss = loss + attr_loss
                    self.metrics_accumulator.update_metric("attr_loss", attr_loss.item())

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    background_loss = torch.zeros([1]).to(self.device)
                    if self.mask is not None:
                        if self.args.attribute and self.mask.mean() > 0.9:
                            # print("Using Effective Attention Maps")
                            masked_background = x_in * (1 - effective_attn)
                            self.attn = effective_attn
                        else:
                            masked_background = x_in * (1 - self.mask)
                    else:
                        masked_background = x_in

                    if self.args.lpips_sim_lambda:
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_image).sum()
                            * self.args.lpips_sim_lambda
                        )
                        background_loss += self.lpips_model(masked_background, self.init_image).sum() * self.args.lpips_sim_lambda
                    if self.args.l2_sim_lambda:
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                        )
                        background_loss += mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                    self.metrics_accumulator.update_metric("background_loss", background_loss.item())

                if self.args.style_lambda != 0:
                    style_base, style_gen = self.init_image * self.mask, x_in * self.mask
                    style_loss = self.style_loss(style_base, style_gen) * self.args.style_lambda # TODO: implement self.style_loss / self.args.style_lambda
                    loss = loss + style_loss
                    self.metrics_accumulator.update_metric("style_loss", style_loss.item())

                return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

            return out

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")

            samples = self.diffusion.p_sample_loop_progressive(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 256
                else {
                    "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                randomize_class=True,
            )

            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        visualization_path = visualization_path.with_stem(
                            f"{visualization_path.stem}_i_{iteration_number}_b_{b}_j_{j}"
                        )

                        # if (
                        #     self.mask is not None
                        #     and self.args.enforce_background
                        #     and j == total_steps
                        #     and not self.args.local_clip_guided_diffusion
                        # ):
                        #     pred_image = (
                        #         self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                        #     )
                        pred_image = (self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0])
                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image_pil = TF.to_pil_image(pred_image)
                        masked_pred_image = self.mask * pred_image.unsqueeze(0)
                        try:
                            final_distance = self.unaugmented_clip_distance(
                                masked_pred_image, text_embed
                            )
                            formatted_distance = f"{final_distance:.4f}"
                        except:
                            formatted_distance = None

                        if self.args.export_assets:
                            pred_path = self.assets_path / visualization_path.name
                            pred_image_pil.save(pred_path)

                        try:
                            if j == total_steps:
                                path_friendly_distance = formatted_distance.replace(".", "")
                                ranked_pred_path = self.ranked_results_path / (
                                    path_friendly_distance + "_" + visualization_path.name
                                )
                                pred_image_pil.save(ranked_pred_path)
                        except:
                            pass

                        intermediate_samples[b].append(pred_image_pil)
                        if should_save_image:
                            if self.args.attribute and self.mask.mean() > 0.9 and self.args.background_preservation_loss:
                                show_editied_masked_image(
                                    title=self.args.prompt,
                                    source_image=self.init_image_pil,
                                    edited_image=pred_image_pil,
                                    mask=TF.to_pil_image(self.attn[0]),
                                    path=visualization_path,
                                    distance=formatted_distance,
                                )
                            else:
                                show_editied_masked_image(
                                    title=self.args.prompt,
                                    source_image=self.init_image_pil,
                                    edited_image=pred_image_pil,
                                    mask=self.mask_pil,
                                    path=visualization_path,
                                    distance=formatted_distance,
                                )

            if self.args.save_video:
                for b in range(self.args.batch_size):
                    video_name = self.args.output_file.replace(
                        ".png", f"_i_{iteration_number}_b_{b}.avi"
                    )
                    video_path = os.path.join(self.args.output_path, video_name)
                    save_video(intermediate_samples[b], video_path)

    def reconstruct_image(self):
        init = Image.open(self.args.init_image).convert("RGB")
        init = init.resize(
            self.image_size,  # type: ignore
            Image.LANCZOS,
        )
        init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)

        samples = self.diffusion.p_sample_loop_progressive(
            self.model,
            (1, 3, self.model_config["image_size"], self.model_config["image_size"],),
            clip_denoised=False,
            model_kwargs={}
            if self.args.model_output_size == 256
            else {"y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)},
            cond_fn=None,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
        save_image_interval = self.diffusion.num_timesteps // 5
        max_iterations = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        for j, sample in enumerate(samples):
            if j % save_image_interval == 0 or j == max_iterations:
                print()
                filename = os.path.join(self.args.output_path, self.args.output_file)
                TF.to_pil_image(sample["pred_xstart"][0].add(1).div(2).clamp(0, 1)).save(filename)

    def preprocess_mask(self):
        image = F.resize(self.init_image, [self.mask_model.image_size, self.mask_model.image_size])
        B = image.size(0)
        image_embeds = self.mask_model.visual_encoder(image)[:, 1:, :] # (B, 576, 768)
        image_embeds = image_embeds.view(B, 24, 24, -1)

    def propose_mask(self):
        """
            model       : BLIP decoder
            query       : target text query
            image       : input image
            n_iter      : number of backprop steps
            lr          : learning rate of backprop
            neg         : whether to use positive log likelihood
            lambda      : weight for target query likelihood
        """

        model = self.mask_model
        query = self.args.prompt
        image = F.resize(self.init_image, [self.mask_model.image_size, self.mask_model.image_size])
        n_iter = self.args.mask_n_iter
        lr = self.args.mask_lr
        neg = self.args.mask_flip
        lambda_ = self.args.mask_lambda

        image_embeds = model.visual_encoder(image)
        image_embeds.requires_grad = True
        image_embeds_ = image_embeds.data.detach().cpu().numpy()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if self.args.mask_base_cap:
            pseudo_cap = self.args.mask_base_cap
        else:
            pseudo_cap = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            # nucleus sampling
            # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
            # print('caption: '+caption[0])
            pseudo_cap = pseudo_cap[0]

        pstext = model.tokenizer(pseudo_cap, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(
            image.device)
        pstext.input_ids[:, 0] = model.tokenizer.bos_token_id

        decoder_pstargets = pstext.input_ids.masked_fill(pstext.input_ids == model.tokenizer.pad_token_id, -100)
        decoder_pstargets[:, :model.prompt_length] = -100

        optimizer = torch.optim.Adam([image_embeds], lr=lr)
        # optimizer = AdamW([image_embeds], lr=lr)

        text = model.tokenizer(query, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(
            image.device)
        text.input_ids[:, 0] = model.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == model.tokenizer.pad_token_id, -100)
        decoder_targets[:, :model.prompt_length] = -100

        num_steps = n_iter

        for _ in range(num_steps):
            decoder_output = model.text_decoder(text.input_ids,
                                                attention_mask=text.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                labels=decoder_targets,
                                                return_dict=True,
                                                )
            decoder_output_base = model.text_decoder(pstext.input_ids,
                                                     attention_mask=pstext.attention_mask,
                                                     encoder_hidden_states=image_embeds,
                                                     encoder_attention_mask=image_atts,
                                                     labels=decoder_pstargets,
                                                     return_dict=True,
                                                     )
            loss_lm = (lambda_ * decoder_output.loss - decoder_output_base.loss)
            if neg:
                loss_lm *= -1
            optimizer.zero_grad()
            loss_lm.backward()
            optimizer.step()

        image_embeds_ = torch.from_numpy(image_embeds_).to(image.device)
        mask = torch.norm(image_embeds - image_embeds_, dim=2)[0][1:].view(24, 24) # TODO: depends on patch embedding size
        # print(mask)

        mask = FF.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            (self.args.model_output_size, self.args.model_output_size),
            mode='bicubic',
            align_corners=False)

        return self.getAttMap(mask.squeeze().detach().cpu().numpy())

    def _normalize(self, x):
        # Normalize to [0, 1].
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    def getAttMap(self, attn_map, blur=False):
        from scipy.ndimage import filters
        if blur:
            attn_map = filters.gaussian_filter(attn_map, 0.02 * max(attn_map.shape[:2]))
            print("after filter ", attn_map.shape)
        attn_map = self._normalize(attn_map)

        # attn_map_c = (attn_map ** 0.7) * attn_map
        # attn_map_c = attn_map ** 2
        return attn_map