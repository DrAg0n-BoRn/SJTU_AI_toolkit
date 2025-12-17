import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from ml_tools.path_manager import list_files_by_extension

from rootpaths import PM


##############################
# reset Hydra instance
GlobalHydra.instance().clear()
initialize_config_dir(config_dir=str(PM.sam_artifacts), version_base='1.2')

# config file name
CONFIG_NAME = 'sam2_1_hiera_l.yaml'

# parameters
MASK_GENERATOR_PARAMS = {
    "points_per_side": 32,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.7,
    "stability_score_thresh": 0.92,
    "stability_score_offset": 0.7,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 0, # if >1 Requires CUDA BUILD,
    "use_m2m": True
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
def build_sam_model(device, postprocessing: bool=True):
    """  
    Returns a built Sam2 model instance
    """
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
def transform_image(path: Path):
    """  
    Opens an image from path and transforms it to numpy array be used in the pipeline.
    """
    image = Image.open(path)
    return np.array(image.convert("RGB"))


# Long process
def generate_mask(mask_generator: SAM2AutomaticMaskGenerator, image: np.ndarray):
    mask_annotations = mask_generator.generate(image)
    
    return mask_annotations


# show mask over image
def render_segmentation(anns: list[dict], original_image: np.ndarray, borders=False, cmap_name="tab20", alpha=0.35, border_color=(1.0, 1.0, 1.0)):
    """
    Overlays segmentation masks on an original image and returns a Pillow Image.
    
    Args:
        anns (list): List of annotation dicts from SAM.
        original_image (np.ndarray): The base image (H, W, 3) or (H, W, 4). 
                                     Assumes RGB (not BGR).
        borders (bool): Whether to draw contours.
        cmap_name (str): Matplotlib colormap name.
        alpha (float): Transparency of the mask fill (0.0 to 1.0).
        border_color (tuple): RGB tuple for borders (0.0 to 1.0).

    Returns:
        PIL.Image: The final composite image in RGBA mode.
    """
    # 1. Ensure input is a valid PIL Image in RGBA mode for compositing
    if isinstance(original_image, np.ndarray):
        # Handle float arrays (0.0-1.0) vs int arrays (0-255)
        if original_image.dtype.kind == 'f':
            original_image = (original_image * 255).astype(np.uint8)
        base_pil = Image.fromarray(original_image).convert("RGBA")
    else:
        raise ValueError("original_image must be a numpy array")

    if not anns:
        return base_pil
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    # Validation check
    if 'segmentation' not in sorted_anns[0]:
        return base_pil

    h, w = sorted_anns[0]['segmentation'].shape
    
    # Check dimensions match
    if base_pil.size != (w, h):
         # Optional: Resize base to match mask or vice versa. 
         # Here we raise to ensure data integrity.
         raise ValueError(f"Image shape {base_pil.size} does not match mask shape {(w, h)}")

    # 2. Generate Mask Layer
    # Initialize transparent layer
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
                contours = [
                    cv2.approxPolyDP(contour, epsilon=0.001 * cv2.arcLength(contour, True), closed=True) 
                    for contour in contours
                ]
                
                border_alpha = min(1.0, alpha + 0.4)
                b_r, b_g, b_b = border_color[:3]
                final_border_color = (b_r, b_g, b_b, border_alpha)

                cv2.drawContours(mask_layer, contours, -1, final_border_color, thickness=1) 

    # 3. Composite using Pillow
    # Convert mask layer from float32 (0-1) to uint8 (0-255)
    mask_uint8 = (mask_layer * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, mode="RGBA")

    # Alpha composite puts the mask_pil 'over' the base_pil
    return Image.alpha_composite(base_pil, mask_pil)


def save_output_img(image: Image.Image, filename: str):
    """
    Saves the PIL Image to the outputs path as a PNG.

    Args:
        image (PIL.Image): The image object.
        filename (str): The base filename.
    """
    if not filename.endswith(".png"):
        filename = filename + ".png"
    
    save_path = PM.sam_outputs / filename

    # Save as PNG to support the RGBA channels
    image.save(save_path, format="PNG")


def check_input_images():
    images_dict: dict[str, Path] = dict()
    
    for valid_extension in VALID_IMG_EXTENSIONS:
        try:
            found_images_dict = list_files_by_extension(directory=PM.sam_inputs, extension=valid_extension, verbose=False)
        except IOError:
            continue
        else:
            print(f"Found {len(found_images_dict)} {valid_extension.upper()} images.")
            images_dict.update(found_images_dict)
    
    # check dict
    if not images_dict:
        raise IOError(f"No valid image files found in directory '{PM.sam_inputs.name}'.")
    
    return images_dict


def main():
    # images
    images_dict = check_input_images()
    # device
    device, dtype = get_device()
    # get model
    sam_model = build_sam_model(device=device, postprocessing=True)
    # generator
    mask_generator = get_generator(sam_model)
    
    count = 0
    # Use inference mode and autocast context
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype):
        for image_name, image_path in images_dict.items():
            try:
                # transform
                image_numpy = transform_image(image_path)
                # generate
                mask_annotations = generate_mask(mask_generator=mask_generator, image=image_numpy)
                # render
                image_rendered = render_segmentation(anns=mask_annotations,
                                                    original_image=image_numpy,
                                                    borders=False)
                # save
                save_output_img(image=image_rendered, filename="sam_" + image_name)
            
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
                
            else:
                count += 1
                if count % 5 == 0:
                    print(f"    > Processed {count}/{len(images_dict)} images...")
    
    print(f"\nSaved {count} SAM2 segmented images to '{PM.sam_outputs.name}'.")


if __name__ == "__main__":
    PM.make_dirs()
    main()
