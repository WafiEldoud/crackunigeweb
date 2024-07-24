import cv2
import os
import numpy as np
import pandas as pd

def reconstruct_image_with_frame(patches, original_shape, results_dataframe, frame_thickness=1):
    reconstructed = np.zeros(original_shape, dtype=np.uint8)
    patch_index = 0
    patch_size = (227,227)
    for y in range(0, original_shape[0], patch_size[0]):
        for x in range(0, original_shape[1], patch_size[1]):
            patch = patches[patch_index]
            PatchIndex = results_dataframe.iloc[patch_index]['Patch Index']
            
            # Find the corresponding row in the DataFrame based on the image name
            row = results_dataframe[results_dataframe['Patch Index'] == PatchIndex].iloc[0]

            if row['Image Classification'] == 'No Crack':
                # Add a white frame to the patch
                patch_with_frame = np.ones_like(patch, dtype=np.uint8) * 255
                patch_with_frame[frame_thickness:patch_size[0]-frame_thickness, 
                                  frame_thickness:patch_size[1]-frame_thickness] = 0  # Make the inner patch black
                
                reconstructed[y:y+patch_size[0], x:x+patch_size[1]] = patch_with_frame
            else:
                # Add a white frame to the patch
                patch_with_frame = np.ones_like(patch, dtype=np.uint8) * 255
                patch_with_frame[frame_thickness:patch_size[0]-frame_thickness, 
                                  frame_thickness:patch_size[1]-frame_thickness] = patch[frame_thickness:patch_size[0]-frame_thickness, 
                                                                                           frame_thickness:patch_size[1]-frame_thickness]
                
                reconstructed[y:y+patch_size[0], x:x+patch_size[1]] = patch_with_frame

            patch_index += 1
            
    return reconstructed

def patchesLabel(patches, results_dataframe):
    patches_with_text = []

    for index, row in results_dataframe.iterrows():
        raw_label_prediction = row['Raw Label Prediction'] * 100  # Multiply by 100

        # Find the corresponding patch in the list of patches
        patch = patches[index]

        # Create a copy of the patch to avoid modifying the original patch
        patch_with_text = patch.copy()

        # Find the center position to place the text
        #text_position = (int(patch_with_text.shape[1] / 2 - len(f"{raw_label_prediction:.2f}") * 5), int(patch_with_text.shape[0] / 2))
        text_position = (35,143)
        # Prepare the text to be displayed
        text = f"{raw_label_prediction:.1f}"

        # Choose text color based on the percentage range
        if 0.0 <= raw_label_prediction <= 20.0:
            text_color = (0, 0, 255)  # Red
        elif 20.0 < raw_label_prediction <= 50.0:
            text_color = (0, 255, 255)  # Yellow
        else:
            text_color = (0, 255, 0)  # Green

        # Increase font size
        font_scale = 2.0
        text_thickness = 3

        # Add the text to the patch with a slight offset to simulate boldness
        for i in range(-1, 2):
            for j in range(-1, 2):
                cv2.putText(patch_with_text, text, (text_position[0] + i, text_position[1] + j), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)

        patches_with_text.append(patch_with_text)

    return patches_with_text

def marked_image(patches, original_shape, results_dataframe, frame_thickness=1):
    reconstructed = np.zeros(original_shape, dtype=np.uint8)
    patch_index = 0
    patch_size = (227,227)
    
    for y in range(0, original_shape[0], patch_size[0]):
        for x in range(0, original_shape[1], patch_size[1]):
            patch = patches[patch_index]
            PatchIndex = results_dataframe.iloc[patch_index]['Patch Index']
            
            # Find the corresponding row in the DataFrame based on the image name
            row = results_dataframe[results_dataframe['Patch Index'] == PatchIndex].iloc[0]

            if row['Image Classification'] == 'No Crack':
                # No Crack patches remain unchanged
                reconstructed[y:y+patch_size[0], x:x+patch_size[1]] = patch
            else:
                # Add a white frame to the patch
                patch_with_frame = np.ones_like(patch, dtype=np.uint8) * 255
                patch_with_frame[frame_thickness:patch_size[0]-frame_thickness, 
                                  frame_thickness:patch_size[1]-frame_thickness] = patch[frame_thickness:patch_size[0]-frame_thickness, 
                                                                                           frame_thickness:patch_size[1]-frame_thickness]

                reconstructed[y:y+patch_size[0], x:x+patch_size[1]] = patch_with_frame

            patch_index += 1
            
    return reconstructed


def patchesLabel2(patches, results_dataframe):
    patches_with_text = []
    previous_row = None

    for index, row in results_dataframe.iterrows():
        raw_label_prediction = row['Raw Label Prediction'] * 100  # Multiply by 100

        # Find the corresponding patch in the list of patches
        patch = patches[index]

        # Create a copy of the patch to avoid modifying the original patch
        patch_with_text = patch.copy()

        # Find the center position to place the text
        text_position = (35, 143)  # Adjust as needed

        # Determine the text to be displayed based on the conditions
        if raw_label_prediction >= 50:
            text = "Crack"
            text_color = (0, 255, 0)  # Red
        else:
            # Check the previous and next rows for raw label prediction
            previous_prediction = previous_row['Raw Label Prediction'] * 100 if previous_row is not None else None
            next_index = index + 1
            next_row = results_dataframe.iloc[next_index] if next_index < len(results_dataframe) else None
            next_prediction = next_row['Raw Label Prediction'] * 100 if next_row is not None else None

            if (previous_prediction is not None and previous_prediction >= 50) and (next_prediction is not None and next_prediction >= 50):
                text = "Possible"
                text_color = (0, 255, 255)  # Yellow
            else:
                text = "No-Crack"
                text_color = (0, 0, 255)  # Green

        # Increase font size
        font_scale = 1.0
        text_thickness = 2

        # Add the text to the patch
        cv2.putText(patch_with_text, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)

        patches_with_text.append(patch_with_text)
        previous_row = row

    return patches_with_text

