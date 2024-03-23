import mygui
import gradcam
import image_a


def main():
    # Initial values
    img_cam = None
    img_cv2 = None
    figure_canvas_agg_grad = None
    toolbar_grad = None
    figure_canvas_agg_image = None
    toolbar_image = None
    figure_canvas_agg_cluster = None
    toolbar_cluster = None
    figure_canvas_agg_target = None
    toolbar_target = None
    processing = False
    gray = None
    gray_standardized = None
    clusters = None
    
    # Window
    window = mygui.main_window()
    
    while True:
        event, values = window.read() # type: ignore
        # print(event, values)
        
        if event == mygui.CLOSED:
            break
        # Load Image
        elif event == "-IMG_PATH-":
            path: str = values[event]
            # Validation
            img_cam, _ = gradcam.read_image_pil(path)
            img_cv2, filename = image_a.read_image_cv(path)
            # Error disable events
            if img_cam is None or img_cv2 is None:
                window.find_element('-WARNING-').update(visible=True)
                window.find_element('-IMG_NAME-').update(value="No Image Selected")
                window.find_element('-IMG_OK-').update(visible=False)
                window.find_element('-IMG_NOTOK-').update(visible=True)
                window.find_element('-GRADCAM-').update(disabled=True, visible=False)
                window.find_element('-IMG_ANALYSIS-').update(disabled=True, visible=False)
            # Img found enable analysis
            else:
                window.find_element('-WARNING-').update(visible=False)
                window.find_element('-IMG_NAME-').update(value=filename)
                window.find_element('-IMG_NOTOK-').update(visible=False)
                window.find_element('-IMG_OK-').update(visible=True)
                window.find_element('-GRADCAM-').update(disabled=False, visible=True)
                if not processing:
                    window.find_element('-IMG_ANALYSIS-').update(disabled=False, visible=True)
                    
        # Grad-CAM
        if event == "-GRADCAM-" and img_cam is not None:
            # Disable button
            window.find_element('-GRADCAM-').update(disabled=True, visible=False)
            # Grad-CAM process
            img_model, img_display = gradcam.transform_image(img_cam)
            model = gradcam.create_model()
            activations, prediction = gradcam.get_gradients(img_model, model)
            heatmap = gradcam.process_heatmap(activations, img_display)
            gradcam_figure = gradcam.plot_gradcam(img_display, heatmap)
            # update message
            window.find_element('-PREDICTED-').update(value=prediction)
            # plot
            figure_canvas_agg_grad, toolbar_grad = mygui.draw_figure(canvas=window.find_element('-GRAD_CANVAS-').TKCanvas, figure=gradcam_figure,     # type: ignore
                                                                    figure_canvas_agg=figure_canvas_agg_grad, toolbar=toolbar_grad) 
        
        # Image Analysis
        if event == "-IMG_ANALYSIS-" and img_cv2 is not None:
            # Disable button and hide tab
            window.find_element('-IMG_ANALYSIS-').update(disabled=True, visible=False)
            window.find_element('-SECRET_TAB-').update(visible=False)
            window.find_element("-TARGET_BTN-").update(visible=False)
            processing = True
            # Image Analysis process
            window.perform_long_operation(lambda: image_a.image_analysis(img_cv2), "-RETURN_TRIGGER-")
        # Long process completed
        elif event == "-RETURN_TRIGGER-":
            mygui.notification_popup()
            processing = False
            # Continue process
            gray, segmented, contrast, gray_standardized = values[event]
            images_figure = image_a.plot_image_analysis(gray, segmented, contrast)
            # plot
            figure_canvas_agg_image, toolbar_image = mygui.draw_figure(canvas=window.find_element('-IMG_CANVAS-').TKCanvas, figure=images_figure,     # type: ignore
                                                                    figure_canvas_agg=figure_canvas_agg_image, toolbar=toolbar_image) 
            # Enable clustering
            window.find_element('-SECRET_TAB-').update(visible=True)
        
        # Clustering
        if event == "-CLUSTER_BTN-":
            # Validate number of clusters
            try:
                n_clusters = int(values['-CLUSTERS-'])
            except ValueError:
                window.find_element("-CLUSTER_ERROR-").update(visible=True)
            else:
                if n_clusters < 2:
                    window.find_element("-CLUSTER_ERROR-").update(visible=True)
                else:
                    window.find_element("-CLUSTER_ERROR-").update(visible=False)
                    # Perform clustering
                    clusters = image_a.image_clustering(gray_standardized, n_clusters)
                    cluster_figure = image_a.plot_image_clustering(clusters, gray)
                    # Plot
                    figure_canvas_agg_cluster, toolbar_cluster = mygui.draw_figure(canvas=window.find_element('-CLUSTER_CANVAS-').TKCanvas, figure=cluster_figure,     # type: ignore
                                                                        figure_canvas_agg=figure_canvas_agg_cluster, toolbar=toolbar_cluster) 
                    # Enable cluster target
                    window.find_element("-TARGET_BTN-").update(visible=True)
                    window.find_element("-TARGET_CLUSTER-").update(values=list(range(0,n_clusters)), value=0)
        # Target cluster
        elif event == "-TARGET_BTN-":
            target = int(values['-TARGET_CLUSTER-'])
            mask_2d, percentage = image_a.target_cluster(clusters, gray, target)
            target_figure = image_a.plot_target_cluster(mask_2d)
            # Plot 
            figure_canvas_agg_target, toolbar_target = mygui.draw_figure(canvas=window.find_element('-TARGET_CANVAS-').TKCanvas, figure=target_figure,     # type: ignore
                                                                        figure_canvas_agg=figure_canvas_agg_target, toolbar=toolbar_target)
            # Update percentage
            window.find_element("-PERCENTAGE-").update(value=f'{percentage}%')

    window.close()


if __name__ == "__main__":
    main()
