import FreeSimpleGUI as sg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk


CLOSED = sg.WIN_CLOSED
CANVAS_SIZE = (1200, 600)
CLUSTER_CANVAS_SIZE = (450, 450)
WINDOW_SIZE = (1350, 850)


# A helper function to plot a figure on a canvas with a toolbar
def draw_figure(canvas, figure, figure_canvas_agg, toolbar):
    # Delete previous figure on canvas
    if figure_canvas_agg:
        figure_canvas_agg.get_tk_widget().forget()
        # Close the OLD figure to free memory
        plt.close(figure_canvas_agg.figure)
        # plt.close(figure)

    # Delete previous toolbar
    if toolbar:
        toolbar.destroy()
        
    # Create canvas
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    # Create toolbar
    toolbar = NavigationToolbar2Tk(figure_canvas_agg, canvas)
    toolbar.update()
    # Pack and return
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1, anchor="center")
    
    return figure_canvas_agg, toolbar


# Create a main window with 2 + 1 tabs
def main_window():
    # settings
    sg.set_options(font=("Helvetica", 16), tooltip_font=("Helvetica", 11))
    sg.theme("Light Green 3")
    
    # Layout
    tab1 = [
        [sg.pin(sg.Button("Predict", key="-GRADCAM-", pad=(20,20), disabled=True, visible=False)), sg.Text(text="Class Predicted:", visible=True, pad=((0,10),(20,20))), 
         sg.Text(text=" ", key="-PREDICTED-", visible=True, relief=sg.RELIEF_SUNKEN, size=(12,1), justification="c", pad=((0,20),(20,20)))],
        [sg.Canvas(size=CANVAS_SIZE, key='-GRAD_CANVAS-')]
    ]
    
    tab2 = [
        [sg.Text(text="Run single-image segmentation using the Segment Anything Model 2 (SAM2).\n\nNote: For batch automation and mask dataset creation, please use the CUDA-enabled Visual-CCC script on GitHub.", pad=(20,20)),
         sg.Button("Run SAM", key="-SAM_BUTTON-", pad=(20,20), visible=False, disabled=True)],
        [sg.Canvas(size=CANVAS_SIZE, key='-SAM_CANVAS-')]
    ]
    
    tab3 = [
        [sg.Text(text="The following visualizations will be generated:\n   1) Image Segmentation\n   2) Texture Analysis\n   3) Clustering Analysis (new tab)", pad=(20,20)), 
         sg.Button("Run Analysis", key="-IMG_ANALYSIS-", pad=(20,20), disabled=True, visible=False),],
        [sg.Canvas(size=CANVAS_SIZE, key='-IMG_CANVAS-')]
    ]
    
    tab4 = [
        [sg.Column([
            [sg.Text(text="Number of clusters:", pad=((20,10),(20,10))), sg.Combo(values=[2,3,4,5,10], default_value=3, key="-CLUSTERS-", pad=((10,20),(20,10)))], 
            [sg.Button('Cluster', key="-CLUSTER_BTN-", pad=((20,20),(20,10)))], 
            [sg.pin(sg.Text(text="INVALID NUMBER!", text_color='red', key="-CLUSTER_ERROR-", visible=False, pad=(20,10))), sg.Text(" ", pad=(20,10))],
            [sg.Canvas(size=CLUSTER_CANVAS_SIZE, key='-CLUSTER_CANVAS-')]
            ]), sg.VerticalSeparator(),
        sg.Column([
            [sg.Text(text="Target cluster:", pad=((20,10),(20,10))), sg.Combo(values=[0], default_value=0, key="-TARGET_CLUSTER-", readonly=True, pad=((10,20),(20,10)))], 
            [sg.pin(sg.Button('Visualize', key="-TARGET_BTN-", pad=((20,20),(20,10)), visible=False)), sg.Text(text=" ", key="-PERCENTAGE-", visible=True, relief=sg.RELIEF_SUNKEN, size=(8,1), justification="c", pad=((20,10),(20,10))),
             sg.Text(text="of the total image area.", pad=((0,10),(20,10)))],
            [sg.Text(" ", pad=(20,10))],
            [sg.Canvas(size=CLUSTER_CANVAS_SIZE, key='-TARGET_CANVAS-')]
            ])
        ]
    ]
    tabgroup = sg.TabGroup(layout=[[sg.Tab(title="Grad-CAM Analysis", layout=tab1, expand_x=True, expand_y=False), 
                                    sg.Tab(title="Segment Anything Model 2", layout=tab2, expand_x=True, expand_y=False),
                                    sg.Tab(title="Image Analysis", layout=tab3, expand_x=True, expand_y=False),
                                    sg.Tab(title="Clustering Analysis", layout=tab4, expand_x=True, expand_y=False, key="-SECRET_TAB-", visible=False)]],
                           expand_x=True, expand_y=True)
    
    layout = [
        [sg.Column([[sg.Image(source=disabled_icon, visible=True, key="-IMG_NOTOK-", subsample=2), sg.Image(source=checkmark_icon, visible=False, key="-IMG_OK-", subsample=2)]]), 
         sg.Text(text="No Image Selected", key="-IMG_NAME-", expand_x=True), sg.Push(),
         sg.Input(disabled=True, key="-IMG_PATH-", visible=False, enable_events=True), sg.FileBrowse(button_text="Select Image", pad=(10,20), key="-PATH_BUTTON-", tooltip="Select Image")],
        [sg.Push(), sg.pin(sg.Text(text="INVALID IMAGE FILE!", key="-WARNING-", visible=False, text_color='red')), sg.Push()],
        [sg.HorizontalSeparator()],
        [tabgroup]
    ]
    
    window = sg.Window(title="VISUAL-CCC", layout=layout, resizable=False, finalize=True, size=WINDOW_SIZE)
    # window.set_min_size(window.size)
    
    return window


def notification_popup():
    sg.popup("Image analysis complete!", title="Process complete", non_blocking=False, keep_on_top=True)

def notification_popup_sam():
    sg.popup("SAM segmentation complete!", title="Process complete", non_blocking=False, keep_on_top=True)

# Base64 images
checkmark_icon = b'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'
disabled_icon = b'iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAFI0lEQVR4nNWbTWhcVRTHfzNOdNEZk2gSd5pS+yEVLGnFYFVEBKFQMSguRCqCuG0aVKhKKbgItbQbP1ZVI3WvZmUXxY920aq0NFWotQRRaGOmEycqtLZmxsWZ4Os4fefc+9678/zDXc2593/OmXPvPfe8ewtkjyHgQeBuYB2wBugH+oByS+ZPoA78BvwAnAG+B74CqgF0TB2bgP3AaaABND1bA5gB9gEjQS3wQAWYQP45X4O19h2wg38jJxfoB3YDNbIzvL1dBHYh06hrKADbgF8JZ3h7qwEvtnQJilXAkQSKp92+BFZmanEETwALGRrj2xaBp7MzW8Jsbw4MjWsNYBKHKWEV7AHeB561DtyG34GjyLT5BvgZWcgWgRKwAhhEwvguYBR4GLjNk+9D4AXgb8/+16AHmMb937jUUmQzcIMHbwG4D3gbvx3mU8S5iVAADjoS14BXgIGk5BGsQHKM8466TJFwh3CZ8w1kmgwmIVRQbul0xUGvSV+ypxxImsB2XyIPjAA/GvVqIDuXE1YhhxMXB9SRORsKN2NfmxaAYevABfyTnDphnVACDhh1+xzjevC8ccC8OKGA3QnqNt4PzBsHy5MTStimwxzQGzfQbsMgLk4YTclACypIMUXT67W4AbSEo4Gs9nUDUTciYSNwVdGpynXqCRNKxyayz4MYVTfId8MJ+w06jXfqqFVyalyb5OTVCRXggqLPTHunTUqHJpLetiOvTnjZoM+GaActbC5x/dw+j04oo9cs3ox2OK0ITymEI9hPbHXC7A7vKnqcXBYcQi9dbzYQ5i0SRhUdlmhF9ZOK4CL283yenFBAT+rGisgXmzgcRbxlwXHgEWT+aegFDpHddGgi+X8c1heBtYrQEUfiE8AWJHI09AKfkV0kHFN+XwfwLfFh8qgneR6mwxaF92uAnxSh1QkU6LYTViucsyDV2Tih3oRKdNMJAwpfFeAvRShxZZXu5Qk3KVyXQXfAjSkp0w0n9Cg8l0GfAremoMgyQk+HPoWjCvoieGdCJdoR0glrlfFni0gExCFtBxwHHsOeJxzC3wnDyu8Xi8BZRUhLlHwQygkbld/PFpE6WhyySlVDpM0PKb+fAf0wNEe2NzCyWhNKyFfpuPHGQMpc2nE466NrFk54QBlnicgON6MIv5XEOiPSzhM+UsY4ERXepwjXkE/UWSOtSBhEkpy4/nuiHUYMhBMpGGhBGk7Yaeh7T3snrS54nnAXFJM4YQi9IHqqE+kOA9nelAy0wNcJUwb57Z0Iy+jngiuEvbPr6oRx9B1tnpj1bJeB6BzJawQucNkdLG1nHFkftquv06RTJ7DCJRK0dayikW0zDnaAsHd003DCMxaiAnL31uqE/0skHHYhWom83rAMPE3YNcHHCTXgdleiMeyvPc4hX5hDYdyoV7Nlw1ZfokkHoqvIV2Z1kUmAIWSfd3mG80YSwgK2xCLa5oCXSDdrHES2L9dr+u+RwkJdAj5xJG62lH0HObX5KFFCjrQH0Q82ndrHGBZoq2LLFxKfc7EggiryofIYUoWZRbLOP5BwriDhfQdwL/LM7n78p9MHyFOaVK7LL6OArAlJnsFl3RrInM80P3mcsC/ErK2OXPAOgmHgiwyM8G2H8djn08BW4BdHZdNsF5DUPfizuSh6gdeRhS6U4fPAq2SbczijjGRoWoE1STuFFDNC1CcTYQNyD+8kUn72NXgJqd7uoUMNLw2EmDsDyL6+HnkStwa4hc7P5xf47/P5WpbK/QNAj1MpTFGL+QAAAABJRU5ErkJggg=='

