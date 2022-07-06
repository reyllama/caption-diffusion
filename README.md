# caption-diffusion

### Requirements

To install dependencies for [Blended Diffusion](https://github.com/omriav/blended-diffusion), run <br>
`$ pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
<br>
We recommend using virtual environments such as conda. <br>

To install dependencies for [BLIP](https://github.com/salesforce/BLIP), run <br>
`$ pip install -r BLIP/requirements.txt` <br>

For automatic mask proposal, you will need a pretrained `BLIP checkpoint` as well as `256x256 imagenet-trained unconditional diffusion model`.
