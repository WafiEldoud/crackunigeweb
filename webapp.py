# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:45:18 2024

@author: nashw
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory

from werkzeug.utils import secure_filename
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from myFunctions.patches import *  # Assuming predict_patches is imported correctly
from myFunctions.reconstruct import *  # Import other necessary functions if needed

model = keras.models.load_model('Xception-NoInput-11April.keras')

app = Flask(__name__, template_folder='')
app.secret_key = '!@3QWeASdZXc'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLOT_FOLDER'] = 'static/plots'

if not os.path.exists(app.config['PLOT_FOLDER']):
    os.makedirs(app.config['PLOT_FOLDER'])

# Function to process image, generate patch filenames, predict patches, and save plot
def process_image(filepath):
    big_image = cv2.imread(filepath)
    patches = extract_flex_patches(big_image)
    big_image_g = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    patches_g = extract_flex_patches(big_image_g)
    height, width, _ = big_image.shape
    
    patch_filenames = []
    for i, patch in enumerate(patches):
        patch_filename = f'patch_{i}.jpg'
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], patch_filename), patch)
        patch_filenames.append(patch_filename)
    
    # Apply prediction on patches
    result_dataframe = predict_patches(patches_g, model)
    data_series = result_dataframe['Raw Label Prediction']
    max_value = data_series.max()
    
    # Plot histogram with a shaded rectangle around crack patches
    plt.figure()  # Create a new figure for the plot
    plt.hist(result_dataframe['Raw Label Prediction'], bins=10)
    plt.xlabel('Prediction')
    plt.ylabel('Frequency')
    plt.axvspan(0.3, 1.0, color='red', alpha=0.3)  # Shaded rectangle around crack patches
    
    threshold = 0.3
    # Calculate the count of data points within the shaded rectangle
    count_in_range = ((data_series >= threshold) & (data_series <= max_value)).sum()

    # Annotate the count within the shaded rectangle
    plt.text((threshold + max_value) / 2, plt.ylim()[1] * 0.9, f'Crack patches: {count_in_range}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Save the generated plot
    plot_filename = 'histogram_plot.png'
    plt.savefig(os.path.join(app.config['PLOT_FOLDER'], plot_filename))
    plt.close()  # Close the plot to release resources
    
    # Reconstruct the image with frame
    original_shape = big_image.shape
    reconstructed_image = reconstruct_image_with_frame(patches, original_shape, result_dataframe)
    reconstructed_image_filename = 'reconstructed_image.png'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], reconstructed_image_filename), reconstructed_image)
    
    return patch_filenames, plot_filename, reconstructed_image_filename


@app.route("/")
@app.route("/Home")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    image_url = session.get('image_url')
    patch_filenames = json.loads(session.get('patch_filenames', '[]'))
    patch_count = len(patch_filenames)
    plot_filename = session.get('plot_filename')
    reconstructed_image_url = session.get('reconstructed_image_url')

    return render_template('about2.html', image_url=image_url, patch_filenames=patch_filenames, patch_count=patch_count, plot_filename=plot_filename, reconstructed_image_url=reconstructed_image_url)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('about'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('about'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['image_url'] = url_for('static', filename='uploads/' + filename)
        
        patch_filenames, plot_filename, reconstructed_image_filename = process_image(filepath)
        
        session['patch_filenames'] = json.dumps(patch_filenames)
        session['plot_filename'] = plot_filename
        session['reconstructed_image_url'] = url_for('static', filename='uploads/' + reconstructed_image_filename)
        
        return redirect(url_for('about'))


# Before request handler to clear session data when navigating away from 'about' page
@app.before_request
def before_request():
    if not request.path.startswith('/about'):
        session.pop('image_url', None)
        session.pop('patch_filenames', None)
        session.pop('plot_filename', None)
        session.pop('reconstructed_image_url', None)

# Route to serve the generated plot
@app.route('/static/plots/<plot_filename>')
def plot_image(plot_filename):
    return send_from_directory(app.config['PLOT_FOLDER'], plot_filename)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)