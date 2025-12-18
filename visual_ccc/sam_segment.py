import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from visual_ccc.paths import PM

##############################
# reset Hydra instance
GlobalHydra.instance().clear()
initialize_config_dir(config_dir=str(PM.resources), version_base='1.2')

# config file name
CONFIG_NAME = 'sam2_1_hiera_l.yaml'

# parameters (default parameters for faster CPU usage)
MASK_GENERATOR_PARAMS = {
    "points_per_side": 32,
    "points_per_batch": 64,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.95,
    "stability_score_offset": 1,
    "mask_threshold": 0,
    "crop_n_layers": 0,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 1,
    "min_mask_region_area": 0, # if >1 Requires CUDA BUILD,
    "use_m2m": False
}

# define valid extensions
VALID_IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'ppm', 'pgm', 'pbm', 'pfm']

#################################

# get the device
def get_device():
    """Selects the best available device and dtype."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Using device: {device}")
    
    return device, dtype


# build sam model
def build_sam_model(device):
    """  
    Returns a built Sam2 model instance
    """
    postprocessing = (device.type == 'cuda')
    
    sam_model = build_sam2(config_file=CONFIG_NAME, 
                    ckpt_path=PM.sam_weights_file,
                    device=device,  # type: ignore
                    mode="eval",
                    apply_postprocessing=postprocessing)
    
    return sam_model


def get_generator(sam_model):
    """ Returns a SAM2 mask generator instance."""
    # mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        **MASK_GENERATOR_PARAMS
    )
    return mask_generator


# transform single image
def transform_image(img: Image.Image):
    """  
    Transforms an image to numpy array to be used in the sam pipeline.
    """
    return np.array(img.convert("RGB"))


# Long process
def generate_mask(mask_generator: SAM2AutomaticMaskGenerator, image: np.ndarray, device, dtype):
    """Long process to generate masks"""
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype):
        mask_annotations = mask_generator.generate(image)
    
    return mask_annotations


# show mask over image
def render_segmentation(anns: list[dict], original_image_array: np.ndarray, borders=False, cmap_name="tab20", alpha=0.35, border_color=(1.0, 1.0, 1.0)):
    """
    Overlays segmentation masks on an original image and returns a Pillow Image and a Matplotlib Figure.
    
    Args:
        anns (list): List of annotation dicts from SAM.
        original_image_array (np.ndarray): The base image (H, W, 3) or (H, W, 4). 
                                     Assumes RGB (not BGR).
        borders (bool): Whether to draw contours.
        cmap_name (str): Matplotlib colormap name.
        alpha (float): Transparency of the mask fill (0.0 to 1.0).
        border_color (tuple): RGB tuple for borders (0.0 to 1.0).

    Returns:
        (PIL Image, Matplotlib Figure): The final composite image in RGBA mode, and a matplotlib figure.
    """
    # 1. Ensure input is a valid PIL Image in RGBA mode
    if isinstance(original_image_array, np.ndarray):
        # Handle float arrays (0.0-1.0) vs int arrays (0-255)
        if original_image_array.dtype.kind == 'f':
            original_image_array = (original_image_array * 255).astype(np.uint8)
        base_pil = Image.fromarray(original_image_array).convert("RGBA")
    else:
        raise ValueError("original_image must be a numpy array")

    # If no annotations, just prepare the base image
    if not anns:
        final_pil = base_pil
    else:
        # Sort anns by area (largest first)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        # Validation check
        if 'segmentation' not in sorted_anns[0]:
            final_pil = base_pil
        else:
            h, w = sorted_anns[0]['segmentation'].shape
            
            # Check dimensions match
            if base_pil.size != (w, h):
                 raise ValueError(f"Image shape {base_pil.size} does not match mask shape {(w, h)}")

            # 2. Generate Mask Layer (using your existing logic)
            mask_layer = np.zeros((h, w, 4), dtype=np.float32)
            cmap = plt.get_cmap(cmap_name)
            num_colors = cmap.N

            for i, ann in enumerate(sorted_anns):
                m = ann['segmentation']
                color = list(cmap(i % num_colors)) 
                color[3] = alpha 
                mask_layer[m] = color 
                
                if borders:
                    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Simplify contours for speed/smoothness
                        contours = [
                            cv2.approxPolyDP(contour, epsilon=0.001 * cv2.arcLength(contour, True), closed=True) 
                            for contour in contours
                        ]
                        border_alpha = min(1.0, alpha + 0.4)
                        b_r, b_g, b_b = border_color[:3]
                        final_border_color = (b_r, b_g, b_b, border_alpha)
                        cv2.drawContours(mask_layer, contours, -1, final_border_color, thickness=1) 

            # 3. Composite using Pillow
            mask_uint8 = (mask_layer * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode="RGBA")
            final_pil = Image.alpha_composite(base_pil, mask_pil)

    # --- Wrap the Result in a Matplotlib Figure ---
    
    # Create a figure, let the GUI canvas (pack fill='both') handle the expansion.
    fig = plt.figure()
    
    # subplot that fills the whole figure
    ax = fig.add_subplot(111)
    
    # Display the composite image
    ax.imshow(final_pil)
    
    # Remove axis ticks and whitespace for a clean "Image-only" look
    ax.axis('off')
    
    # Tight_layout removes the default white margins Matplotlib adds
    fig.tight_layout(pad=0)
    
    return final_pil, fig
