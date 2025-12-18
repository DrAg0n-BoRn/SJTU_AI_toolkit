import numpy
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
from joblib import Parallel, delayed


# Settings
OUTPUT_SIZE = 512
FIGSIZE_L = (10, 5)
FIGSIZE_S = (4, 4)
DPI = 150
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm', '.pgm', '.pbm', '.pfm']
# Create a custom colormap
my_colors = ['#FFFFFF', '#0000FF']
my_cmap = mcolors.LinearSegmentedColormap.from_list('Custom', my_colors, N=256)


# Read image CV2
def read_image_cv(path):
    # Validation
    if not os.path.isfile(path):
        return None, None
    # Check if file is an image (by extension)
    _, extension = os.path.splitext(path)
    if extension.lower() not in valid_extensions:
        return None, None
    
    filename: str = os.path.basename(path)
    original_img = cv2.imread(path)
    return original_img, filename


# Image Analysis: Grayscale - Image segmentation - Texture
def image_segmentation(img, threshold: int=127):
    # Resize image
    ratio = OUTPUT_SIZE / img.shape[1]
    dim = (OUTPUT_SIZE, int(img.shape[0] * ratio))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # Convert the image to true grayscale
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Perform image segmentation (Otsu's Binarization)
    # Otsu's method finds the best value and return it in _thresh.
    _thresh, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return gray, segmented


# Helper function to compute contrast for a single ROW (for parallel processing)
def _compute_contrast_row(segmented, i, window_size, cols):
    ws_h = window_size // 2
    contrast_row = numpy.zeros((cols,), dtype=float)
    
    for j in range(ws_h, cols - ws_h):
        # Extract window
        window = segmented[i-ws_h : i+ws_h+1, j-ws_h : j+ws_h+1]
        
        # Scale to 0-1 and use levels=2
        window_norm = (window // 255).astype(numpy.uint8)
        
        # Calculate GLCM on the 2-level normalized window
        glcm_window = graycomatrix(window_norm, [1], [0], levels=2, symmetric=True, normed=True)
        contrast_window = graycoprops(glcm_window, 'contrast')
        contrast_row[j] = contrast_window[0,0]
        
    return i, contrast_row


# Calculate GLCM (Gray Level Co-occurrence Matrix) to get texture features
def image_texture(segmented):
    contrast = numpy.zeros_like(segmented, dtype=float)
    window_size = 5
    ws_h = window_size // 2
    rows, cols = segmented.shape

    # Create a list of all rows to process
    row_indices = range(ws_h, rows - ws_h)

    # Run in parallel (n_jobs=-1 uses all available cores) (use threads to avoid freeze bombs on deployment)
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_contrast_row)(segmented, i, window_size, cols) for i in row_indices
    )

    # Reconstruct the contrast image from results
    for i, c_row in results: # type: ignore
        contrast[i, :] = c_row
    
    return contrast


# standardize image pixel values
def standardize_image(gray):
    # Standardize values in gray
    scaler = StandardScaler()
    gray_standardized = scaler.fit_transform(gray.reshape(-1,1))
    
    return gray_standardized


# Plot gray, segmented and texture
def plot_image_analysis(gray, segmented, contrast):
    # Figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE_L, dpi=DPI) 
    
    # show grayscale image
    axs[0].imshow(gray, cmap='gray')
    axs[0].title.set_text('Grayscale')
    axs[0].axis('off')
    
    # show segmented image
    axs[1].imshow(segmented, cmap='gray')
    axs[1].title.set_text('Image Segmentation')
    axs[1].axis('off')
    
    # show texture (contrast)
    axs[2].imshow(contrast, cmap='hot')
    axs[2].title.set_text('Texture Analysis')
    axs[2].axis('off')
    
    # Adjust subplot params to reduce padding
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.1)
    
    return fig


# Image clustering
def image_clustering(gray_standardized, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(gray_standardized)
    
    return clusters


# Plot image clustering
def plot_image_clustering(clusters, gray):
    fig = plt.figure(figsize=FIGSIZE_S, dpi=DPI)
    axs = fig.add_axes((0,0,1,1))
    axs.set_title("Clustering Result")
    axs.imshow(clusters.reshape(gray.shape), cmap='viridis', label="Clusters")
    axs.set_axis_off()
    
    # Create handles and labels
    handles = []
    labels = []
    # Loop through the unique values in the clusters array
    for i in numpy.unique(clusters):
        # Create a patch with the color of the cluster
        patch = mpatches.Patch(color=plt.cm.viridis(i / clusters.max())) # type: ignore
        # Append the patch and the cluster number to the lists
        handles.append(patch)
        labels.append(f'{i}')
    # Create a legend for the image using the handles and labels
    axs.legend(handles, labels, loc='upper right', title='Clusters')
    
    return fig


# Get pixels in target cluster
def target_cluster(clusters, gray, target):
    # Create a mask for the cluster
    mask = clusters == target

    # Reshape the mask to have the same shape as original image
    mask_2d = mask.reshape(gray.shape)

    # Store pixels
    cluster_pixels = gray[mask_2d]
    
    # Cluster percentage size 
    percentage = round(100 * cluster_pixels.flatten().shape[0] / gray.flatten().shape[0], 2)
    
    return mask_2d, percentage


# Plot target cluster
def plot_target_cluster(mask_2d):
    fig = plt.figure(figsize=FIGSIZE_S, dpi=DPI)
    axs = fig.add_axes((0,0,1,1))
    axs.set_title("Target Cluster")
    axs.imshow(mask_2d, cmap=my_cmap)
    axs.set_axis_off()
    
    return fig
