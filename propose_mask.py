import sys
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_image(image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    w, h = raw_image.size
    display(raw_image.resize((w // 5, h // 5)))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def propose_mask(query, image):
    image_embeds = model.visual_encoder(image).data
    image_embeds.requires_grad = True
    image_embeds_ = image_embeds.data.detach().cpu().numpy()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    optimizer = torch.optim.Adam([image_embeds], lr=0.05)

    query = "a woman and her bear on the beach"
    text = model.tokenizer(query, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
    text.input_ids[:, 0] = model.tokenizer.bos_token_id

    decoder_targets = text.input_ids.masked_fill(text.input_ids == model.tokenizer.pad_token_id, -100)
    decoder_targets[:, :model.prompt_length] = -100

    num_steps = 3

    for it in tqdm(range(num_steps)):
        decoder_output = model.text_decoder(text.input_ids,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        labels = decoder_targets,
                                        return_dict = True,
                                        )
        loss_lm = decoder_output.loss
        optimizer.zero_grad()
        # grad = torch.autograd.grad(outputs=loss_lm, inputs=image_embeds, create_graph=True)
        # print(grad)
        loss_lm.backward()
        optimizer.step()

    image_embeds_ = torch.from_numpy(image_embeds_).to(image.device)
    torch.sum(image_embeds_ - image_embeds)
    grids = torch.norm(image_embeds-image_embeds_, dim=2)[0][1:].view(24, 24)

    grids = F.interpolate(
            grids.unsqueeze(0).unsqueeze(0),
            image.shape[2:],
            mode='bicubic',
            align_corners=False)

    viz_attn(image.squeeze().permute(1,2,0).detach().cpu().numpy(), grids.squeeze().detach().cpu().numpy(), True)