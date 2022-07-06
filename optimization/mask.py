import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import tqdm

def normalize(x):
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1 * (1 - attn_map ** 0.7).reshape(attn_map.shape + (1,)) * img + \
               (attn_map ** 0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, attn_map2, blur=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    axes[2].imshow(getAttMap(img, attn_map2, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

def propose_mask(model, query, image, n_iter=3, lr=0.05, neg=False, lambda_=1.0):
    image_embeds = model.visual_encoder(image)
    image_embeds.requires_grad = True
    image_embeds_ = image_embeds.data.detach().cpu().numpy()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    pseudo_cap = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    # print('caption: '+caption[0])
    pseudo_cap = pseudo_cap[0]

    pstext = model.tokenizer(pseudo_cap, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
    pstext.input_ids[:, 0] = model.tokenizer.bos_token_id

    decoder_pstargets = pstext.input_ids.masked_fill(pstext.input_ids == model.tokenizer.pad_token_id, -100)
    decoder_pstargets[:, :model.prompt_length] = -100

    vals = torch.norm(image_embeds, dim=2).detach().cpu().data[0][1:].view(24,24)

    grid_vals = F.interpolate(
            vals.unsqueeze(0).unsqueeze(0),
            image.shape[2:],
            mode='bicubic',
            align_corners=False)

    optimizer = torch.optim.Adam([image_embeds], lr=lr)
    # optimizer = AdamW([image_embeds], lr=lr)

    text = model.tokenizer(query, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
    text.input_ids[:, 0] = model.tokenizer.bos_token_id

    decoder_targets = text.input_ids.masked_fill(text.input_ids == model.tokenizer.pad_token_id, -100)
    decoder_targets[:, :model.prompt_length] = -100

    num_steps = n_iter

    for it in tqdm(range(num_steps)):
        decoder_output = model.text_decoder(text.input_ids,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        labels = decoder_targets,
                                        return_dict = True,
                                        )
        decoder_output_base = model.text_decoder(pstext.input_ids,
                                        attention_mask = pstext.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        labels = decoder_pstargets,
                                        return_dict = True,
                                        )
        loss_lm = (lambda_ * decoder_output.loss - decoder_output_base.loss)
        if neg:
            loss_lm *= -1
        optimizer.zero_grad()
        # grad = torch.autograd.grad(outputs=loss_lm, inputs=image_embeds, create_graph=True)
        # print(grad)
        loss_lm.backward()
        optimizer.step()

    image_embeds_ = torch.from_numpy(image_embeds_).to(image.device)
    grids = torch.norm(image_embeds-image_embeds_, dim=2)[0][1:].view(24, 24)

    grids = F.interpolate(
            grids.unsqueeze(0).unsqueeze(0),
            image.shape[2:],
            mode='bicubic',
            align_corners=False)

    viz_attn(image.squeeze().permute(1,2,0).detach().cpu().numpy(), grid_vals.squeeze().detach().cpu().numpy(), grids.squeeze().detach().cpu().numpy(), True)