# caption-diffusion

### Requirements

To install dependencies for [Blended Diffusion](https://github.com/omriav/blended-diffusion), run <br>
`$ pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
<br>
We recommend using virtual environments such as conda. <br>

To install dependencies for [BLIP](https://github.com/salesforce/BLIP), run <br>
`$ pip install -r BLIP/requirements.txt` <br>

For automatic mask proposal, you will need a pretrained `BLIP checkpoint` as well as `256x256 imagenet-trained unconditional diffusion model`. <br>
Note that BLIP decoder checkpoint is downloaded online.

<br>
Auto-mask running script example: <br>
`$ python3 main.py -p 'v-neck t shirts' -i 'validation/fashion-shop/basic_crew_neck_tee.jpg' --mask_auto --output_path 'output/' --vit --mask_thresh 0.35 --style_lambda 0.01 --ot`
<br>

### Hyperparameters

Name | Role | type
---- | ---- | ----
mask_auto     | whether to automatically generate mask; you need manual mask otherwise with `--mask`  | `store_true`
mask_n_iter   | how much iterations to run backprops for mask generation                              | `int`
mask_lr       | learning rate for mask backprop                                                       | `float`
mask_flip     | whether to flip signs for maximum likelihood during mask proposal                     | `store_true`
mask_lambda   | relative loss weight for pseudo capion and target caption                             | `float`
mask_thresh   | threshold for binarizing the proposed mask                                            | `float`
mask_base_cap | manual base caption input; we do not use BLIP pseudo caption                          | `str`
vit           | whether to use BLIP ViT encoder for style loss; we use VGG otherwise                  | `store_true`
pseudo_cap    | whether to generate pseudo caption for loss guidance (vector difference)              | `store_true`
blur          | whether to apply Gaussian blur to the proposed mask                                   | `store_ture`
ot            | whether to use optimal transport (feature l2) instead of Gram Matrix style loss       | `store_true`
