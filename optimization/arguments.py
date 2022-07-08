import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "-p", "--prompt", type=str, help="The prompt for the desired editing", required=True, default=None
    )
    parser.add_argument(
        "-i", "--init_image", type=str, help="The path to the source image input", required=True
    )
    # parser.add_argument(
    #     "-a", "--attribute", type=str, help="The attributes for the image separated by white space", required=False, default=None
    # )
    parser.add_argument("--mask", type=str, help="The path to the mask to edit with", default=None)

    # Diffusion
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=25,
    )
    parser.add_argument(
        "--guidance",
        default="clip",
        choices=["clip", "blip"],
        help="how to provide diffusion guidance"
    )
    parser.add_argument(
        "--local_clip_guided_diffusion",
        help="Indicator for using local CLIP guided diffusion (for baseline comparison)",
        action="store_true",
        dest="local_clip_guided_diffusion",
    )

    # For more details read guided-diffusion/guided_diffusion/respace.py
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="100",
    )
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )

    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)

    # Loss
    parser.add_argument(
        "--guidance_lambda",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=1000,
    )
    parser.add_argument(
        "--attribute_guidance_lambda",
        type=float,
        help="Controls how much the image should look like the attributes",
        default=10.0,
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=50,
    )
    parser.add_argument(
        "--lpips_sim_lambda",
        type=float,
        help="The LPIPS similarity to the input image",
        default=1000,
    )
    parser.add_argument(
        "--l2_sim_lambda", type=float, help="The L2 similarity to the input image", default=10000,
    )
    parser.add_argument(
        "--background_preservation_loss",
        help="Indicator for using the background preservation loss",
        action="store_true",
    )
    parser.add_argument(
        "--style_lambda",
        type=float,
        help="style preservation loss weight",
        default=1.0
    )
    parser.add_argument("--finetuned", action='store_true', help='whether to load BLIP ckpt finetuned on COCO captioning')
    parser.add_argument("--vit", action='store_true')
    parser.add_argument("--pseudo_cap", action="store_true")
    parser.add_argument("--blur", action='store_true', help='whether to blur the mask (after binarization)')
    parser.add_argument("--ot", action='store_true', help='whether to employ feature matching loss (optimal transport)')

    # Mask
    parser.add_argument(
        "--invert_mask",
        help="Indicator for mask inversion",
        action="store_true",
        dest="invert_mask",
    )
    parser.add_argument(
        "--no_enforce_background",
        help="Indicator disabling the last background enforcement",
        action="store_false",
        dest="enforce_background",
    )
    parser.add_argument("--mask_auto", action='store_true', help='whether to automatically generate mask')
    parser.add_argument("--mask_n_iter", type=int, default=15)
    parser.add_argument("--mask_lr", type=float, default=0.01)
    parser.add_argument("--mask_flip", action='store_true', help='whether to flip signs for maximum log likelihood')
    parser.add_argument("--mask_lambda", type=float, default=1.0)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--mask_base_cap", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, help="The random seed", default=404)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=8)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=4,
    )
    parser.add_argument(
        "--vid",
        help="Indicator for saving the video of the diffusion process",
        action="store_true",
        dest="save_video",
    )
    parser.add_argument(
        "--export_assets",
        help="Indicator for saving raw assets of the prediction",
        action="store_true",
        dest="export_assets",
    )

    args = parser.parse_args()
    return args
