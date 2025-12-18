from visual_ccc import mygui
from visual_ccc import gradcam
from visual_ccc import image_a
from visual_ccc import sam_segment
import multiprocessing


def main():
    # Initial values
    img_original_pil = None
    img_cv2 = None
    img_sam = None
    original_img_array = None
    figure_canvas_agg_grad = None
    toolbar_grad = None
    figure_canvas_agg_sam = None
    toolbar_sam = None
    figure_canvas_agg_image = None
    toolbar_image = None
    figure_canvas_agg_cluster = None
    toolbar_cluster = None
    figure_canvas_agg_target = None
    toolbar_target = None
    is_img_processed = False
    is_sam_processed = False
    processing = False
    gray = None
    gray_standardized = None
    clusters = None
    # Load model in evaluation mode
    model, class_map = gradcam.create_model()
    model.cpu()
    model.eval()
    # Setup SAM
    sam_device, sam_dtype = sam_segment.get_device()
    sam_model = sam_segment.build_sam_model(device=sam_device)
    sam_model.to(sam_device)
    sam_model.eval()
    sam_mask_generator = sam_segment.get_generator(sam_model)
    
    # validate class map
    if class_map is None or not isinstance(class_map, dict):
        class_map = {"0": 0,
                     "1": 1,
                     "2": 2}
    
    # Window
    window = mygui.main_window()
    
    while True:
        event, values = window.read() # type: ignore
        # print(event, values)
        
        if event == mygui.CLOSED:
            break
        # Load Image
        elif event == "-IMG_PATH-" and not processing:
            path: str = values[event]
            # Validation
            img_original_pil, _ = gradcam.read_image_pil(path)
            img_cv2, filename = image_a.read_image_cv(path)
            if img_original_pil:
                img_sam = sam_segment.transform_image(img_original_pil)
            # Error disable events
            if img_original_pil is None or img_cv2 is None:
                window.find_element('-WARNING-').update(visible=True) # type: ignore
                window.find_element('-IMG_NAME-').update(value="No Image Selected") # type: ignore
                window.find_element('-IMG_OK-').update(visible=False) # type: ignore
                window.find_element('-IMG_NOTOK-').update(visible=True) # type: ignore
                window.find_element('-GRADCAM-').update(disabled=True, visible=False) # type: ignore
                window.find_element('-IMG_ANALYSIS-').update(disabled=True, visible=False) # type: ignore
                window.find_element('-SAM_BUTTON-').update(disabled=True, visible=False) # type: ignore
            # Img found enable analysis
            else:
                window.find_element('-WARNING-').update(visible=False) # type: ignore
                window.find_element('-IMG_NAME-').update(value=filename) # type: ignore
                window.find_element('-IMG_NOTOK-').update(visible=False) # type: ignore
                window.find_element('-IMG_OK-').update(visible=True) # type: ignore
                window.find_element('-GRADCAM-').update(disabled=False, visible=True) # type: ignore

                window.find_element('-IMG_ANALYSIS-').update(disabled=False, visible=True) # type: ignore
                window.find_element('-SAM_BUTTON-').update(disabled=False, visible=True) # type: ignore
                is_img_processed = False
                is_sam_processed = False
                    
        # Grad-CAM
        if event == "-GRADCAM-" and img_original_pil is not None:
            # Disable button
            window.find_element('-GRADCAM-').update(disabled=True, visible=False) # type: ignore
            # Grad-CAM process
            img_model, img_display = gradcam.transform_image(img_original_pil)
            # CHOOSE BINARY OR TERNARY MODEL
            activations, prediction = gradcam.get_gradients_multiclass(img_model, model, class_map)
            heatmap = gradcam.process_heatmap(activations, img_display)
            gradcam_figure = gradcam.plot_gradcam(img_display, heatmap)
            # update message
            window.find_element('-PREDICTED-').update(value=prediction) # type: ignore
            # plot
            figure_canvas_agg_grad, toolbar_grad = mygui.draw_figure(canvas=window.find_element('-GRAD_CANVAS-').TKCanvas, figure=gradcam_figure,     # type: ignore
                                                                    figure_canvas_agg=figure_canvas_agg_grad, toolbar=toolbar_grad) 
        
        # SAM
        if event == "-SAM_BUTTON-" and img_sam is not None:
            # Disable items
            window.find_element('-IMG_ANALYSIS-').update(disabled=True, visible=False) # type: ignore
            window.find_element('-SAM_BUTTON-').update(disabled=True, visible=False) # type: ignore
            window.find_element('-PATH_BUTTON-').update(disabled=True, tooltip="Disabled while processing image") # type: ignore
            processing = True
            # save original image
            original_img_array = img_sam.copy()
            # SAM process
            window.perform_long_operation(lambda: sam_segment.generate_mask(mask_generator=sam_mask_generator,
                                                                            image=original_img_array, # type: ignore
                                                                            device=sam_device,
                                                                            dtype=sam_dtype), 
                                          "-RETURN_SAM_TRIGGER-")
        elif event == "-RETURN_SAM_TRIGGER-" and original_img_array:
            mygui.notification_popup_sam()
            mask_annotations = values[event]
            # get rendered image
            _pil_image, mat_sam_figure = sam_segment.render_segmentation(anns=mask_annotations, original_image_array=original_img_array, borders=False)
            # plot
            figure_canvas_agg_sam, toolbar_sam = mygui.draw_figure(canvas=window.find_element('-SAM_CANVAS-').TKCanvas, figure=mat_sam_figure,     # type: ignore
                                                                    figure_canvas_agg=figure_canvas_agg_sam, toolbar=toolbar_sam)
            processing = False
            is_sam_processed = True
            # restore buttons
            window.find_element('-PATH_BUTTON-').update(disabled=False, tooltip=None) # type: ignore
            if not is_img_processed:
                window.find_element('-IMG_ANALYSIS-').update(disabled=False, visible=True) # type: ignore
        
        # Image Analysis
        if event == "-IMG_ANALYSIS-" and img_cv2 is not None:
            # Disable button and hide tab
            window.find_element('-IMG_ANALYSIS-').update(disabled=True, visible=False) # type: ignore
            window.find_element('-SAM_BUTTON-').update(disabled=True, visible=False) # type: ignore
            window.find_element('-SECRET_TAB-').update(visible=False) # type: ignore
            window.find_element("-TARGET_BTN-").update(visible=False) # type: ignore
            window.find_element('-PATH_BUTTON-').update(disabled=True, tooltip="Disabled while processing image") # type: ignore
            processing = True
            # Image Analysis process
            gray, segmented = image_a.image_segmentation(img_cv2)
            window.perform_long_operation(lambda: image_a.image_texture(segmented), "-RETURN_TRIGGER-")
        # Long process completed
        elif event == "-RETURN_TRIGGER-":
            mygui.notification_popup()
            # Continue process
            contrast = values[event]
            images_figure = image_a.plot_image_analysis(gray, segmented, contrast)
            # save standardized version for later use in clustering
            gray_standardized = image_a.standardize_image(gray)
            # plot
            figure_canvas_agg_image, toolbar_image = mygui.draw_figure(canvas=window.find_element('-IMG_CANVAS-').TKCanvas, figure=images_figure,     # type: ignore
                                                                    figure_canvas_agg=figure_canvas_agg_image, toolbar=toolbar_image) 
            # Enable clustering
            window.find_element('-SECRET_TAB-').update(visible=True) # type: ignore
            # Restore buttons
            window.find_element('-PATH_BUTTON-').update(disabled=False, tooltip=None) # type: ignore
            processing = False
            is_img_processed = True
            if not is_sam_processed:
                window.find_element('-SAM_BUTTON-').update(disabled=False, visible=True) # type: ignore
        
        # Clustering
        if event == "-CLUSTER_BTN-":
            # Validate number of clusters
            try:
                n_clusters = int(values['-CLUSTERS-'])
            except ValueError:
                window.find_element("-CLUSTER_ERROR-").update(visible=True) # type: ignore
            else:
                if n_clusters < 2:
                    window.find_element("-CLUSTER_ERROR-").update(visible=True) # type: ignore
                else:
                    window.find_element("-CLUSTER_ERROR-").update(visible=False) # type: ignore
                    # Perform clustering
                    clusters = image_a.image_clustering(gray_standardized, n_clusters)
                    cluster_figure = image_a.plot_image_clustering(clusters, gray)
                    # Plot
                    figure_canvas_agg_cluster, toolbar_cluster = mygui.draw_figure(canvas=window.find_element('-CLUSTER_CANVAS-').TKCanvas, figure=cluster_figure,     # type: ignore
                                                                        figure_canvas_agg=figure_canvas_agg_cluster, toolbar=toolbar_cluster) 
                    # Enable cluster target
                    window.find_element("-TARGET_BTN-").update(visible=True) # type: ignore
                    window.find_element("-TARGET_CLUSTER-").update(values=list(range(0,n_clusters)), value=0) # type: ignore
        # Target cluster
        elif event == "-TARGET_BTN-":
            target = int(values['-TARGET_CLUSTER-'])
            mask_2d, percentage = image_a.target_cluster(clusters, gray, target)
            target_figure = image_a.plot_target_cluster(mask_2d)
            # Plot 
            figure_canvas_agg_target, toolbar_target = mygui.draw_figure(canvas=window.find_element('-TARGET_CANVAS-').TKCanvas, figure=target_figure,     # type: ignore
                                                                        figure_canvas_agg=figure_canvas_agg_target, toolbar=toolbar_target)
            # Update percentage
            window.find_element("-PERCENTAGE-").update(value=f'{percentage}%') # type: ignore

    window.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
