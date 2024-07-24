import numpy as np
import cv2
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt

def extract_flex_patches(image):
    patches = []
    height, width = image.shape[:2]

    # Define the standard patch size
    patch_size = (227, 227)

    # Calculate the number of patches along each dimension
    num_patches_height = height // patch_size[1]
    num_patches_width = width // patch_size[0]

    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            # Adjust patch size if necessary to fit exactly within the image
            patch_height = min(patch_size[1], height - y)
            patch_width = min(patch_size[0], width - x)
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)

    return patches


def create_data3(folder_path, img_file):
    bi_inv_data=[] # Binary inversed - cracks in white
    dir_ = os.path.join(folder_path, img_file)
    image = cv2.imread(dir_, cv2.IMREAD_GRAYSCALE) # 0 reads image in grayscale 0-255
    height, width = image.shape[:2]  # Get the height and width of the original image
    image_rescaled = image / 255.0  # Rescale the image to values between 0 and 1
    bi_inv = image_rescaled # returns binary inversed image-white cracks +  original image
    bi_inv_data.append(bi_inv)   

    return bi_inv_data, height, width

def create_data_from_patch(patch):
    bi_inv_data = []  # Binary inversed - cracks in white
    height, width = patch.shape[:2]  # Get the height and width of the patch
    patch_rescaled = patch / 255.0  # Rescale the patch to values between 0 and 1
    bi_inv = patch_rescaled  # returns binary inversed patch - white cracks + original patch
    bi_inv_data.append(bi_inv)   

    return bi_inv_data, height, width

def predict_image_util(final_pred_inv, height, width, model):
    img_test = np.array(final_pred_inv).reshape((1, height, width, 1))
    raw_predicted_label = model.predict(img_test, batch_size=None, verbose=0, steps=None)[0][0]

    predicted_label = 1
    if raw_predicted_label < 0.3:
        predicted_label = 0

    predicted_label_str = 'Crack' if predicted_label == 1 else 'No Crack'

    return raw_predicted_label, predicted_label_str

def predict_folder(folder_path, model):
    image_names = []
    raw_predictions = []
    image_classifications = []

    for img_file in os.listdir(folder_path):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(folder_path, img_file)
            image_names.append(img_file)

            pred_data_inv_, height, width = create_data3(folder_path, img_file)
            pred_data_inv = []
            pred_data_inv.append(pred_data_inv_[0])
            final_pred_inv = np.array(pred_data_inv)
            raw_pred, img_classification = predict_image_util(final_pred_inv, height, width, model)
            raw_predictions.append(raw_pred)
            image_classifications.append(img_classification)

    data = {
        'Image Name': image_names,
        'Raw Label Prediction': raw_predictions,
        'Image Classification': image_classifications
    }

    df = pd.DataFrame(data)
    df['Length'] = df['Image Name'].apply(len)

    df_sorted = df.sort_values(by=['Length', 'Image Name']).drop('Length', axis=1)
    df_sorted.reset_index(drop=True, inplace=True)
    return df_sorted

def predict_patches(patches, model):
    raw_predictions = []
    image_classifications = []

    for patch in patches:
        pred_data_inv_, height, width = create_data_from_patch(patch)
        final_pred_inv = np.array(pred_data_inv_)
        raw_pred, img_classification = predict_image_util(final_pred_inv, height, width, model)
        raw_predictions.append(raw_pred)
        image_classifications.append(img_classification)

    data = {
        'Patch Index': list(range(len(patches))),
        'Raw Label Prediction': raw_predictions,
        'Image Classification': image_classifications
    }

    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='Patch Index')
    df_sorted.reset_index(drop=True, inplace=True)
    return df_sorted

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(folder_path)


def plot_histogram_with_threshold(data_series, threshold):
    plt.figure(figsize=(8, 6))
    plt.hist(data_series, bins=10, edgecolor='black')
    plt.xlabel('Raw Label Prediction')
    plt.ylabel('Frequency')
    plt.title('Frequency of crack patches')

    # Draw a rectangle starting from the threshold value (0.5) to the maximum value in data_series
    max_value = data_series.max()
    plt.axvspan(threshold, max_value, color='red', alpha=0.3)  # Highlight in red with transparency

    # Calculate the count of data points within the shaded rectangle
    count_in_range = ((data_series >= threshold) & (data_series <= max_value)).sum()

    # Annotate the count within the shaded rectangle
    plt.text((threshold + max_value) / 2, plt.ylim()[1] * 0.9, f'Crack patches: {count_in_range}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.grid(False)
    plt.show()


def reconstruct_image_with_frame(patches, original_shape, patch_size, results_dataframe, frame_thickness=1):
    reconstructed = np.zeros(original_shape, dtype=np.uint8)
    patch_index = 0
    
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