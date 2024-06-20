from flask import Flask, request, render_template, send_from_directory, url_for
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

def compute_snr(original_data, processed_data):
    signal_power = np.mean(original_data ** 2)
    noise_power = np.mean((original_data - processed_data) ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')

def compute_psnr(original_data, processed_data):
    mse = np.mean((original_data - processed_data) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))

def apply_kmeans_compression(image_data, compression_level):
    k_values = {'low': 8, 'medium': 16, 'high': 32}
    k = k_values[compression_level]
    original_shape = image_data.shape
    img_data_flattened = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data_flattened)
    compressed_img = kmeans.cluster_centers_[kmeans.labels_]
    compressed_img = compressed_img.reshape(original_shape).astype(np.uint8)
    return compressed_img, compute_snr(image_data, compressed_img), compute_psnr(image_data, compressed_img)

def apply_noise_reduction(image_data, reduction_type, level):
    ksize_values = {'low': 3, 'medium': 5, 'high': 7}
    ksize = ksize_values[level]
    if reduction_type == 'median':
        reduced_img = cv2.medianBlur(image_data, ksize)
    else:  # gaussian
        reduced_img = cv2.GaussianBlur(image_data, (ksize, ksize), 0)
    return reduced_img, compute_snr(image_data, reduced_img), compute_psnr(image_data, reduced_img)

def apply_image_sharpening(image_data, level):
    strength = {'low': -1, 'medium': -3, 'high': -5}
    kernel = np.array([[0, strength[level], 0],
                       [strength[level], 9 - 4*strength[level], strength[level]],
                       [0, strength[level], 0]])
    sharpened_img = cv2.filter2D(image_data, -1, kernel)
    return sharpened_img, compute_snr(image_data, sharpened_img), compute_psnr(image_data, sharpened_img)

def apply_segmentation(image_data, level):
    k_values = {'low': 2, 'medium': 4, 'high': 8}
    k = k_values[level]
    original_shape = image_data.shape
    img_data_flattened = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data_flattened)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(original_shape).astype(np.uint8)
    return segmented_img, compute_snr(image_data, segmented_img), compute_psnr(image_data, segmented_img)

def convert_to_grayscale(image_data):
    grayscale_img = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    grayscale_img_rgb = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)  # To keep the dimensions consistent
    return grayscale_img_rgb, compute_snr(image_data, grayscale_img_rgb), compute_psnr(image_data, grayscale_img_rgb)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        operation = request.form.get('operation')
        level = request.form.get('level', 'low')
        
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = Image.open(filepath)
        img_data = np.array(img)
        processed_img_data, snr, psnr = None, None, None

        if operation == 'compress':
            processed_img_data, snr, psnr = apply_kmeans_compression(img_data, level)
        elif operation == 'reduce_noise':
            noise_type = request.form.get('noise_type', 'median')
            processed_img_data, snr, psnr = apply_noise_reduction(img_data, noise_type, level)
        elif operation == 'sharpen':
            processed_img_data, snr, psnr = apply_image_sharpening(img_data, level)
        elif operation == 'segment':
            processed_img_data, snr, psnr = apply_segmentation(img_data, level)
        elif operation == 'grayscale':
            processed_img_data, snr, psnr = convert_to_grayscale(img_data)

        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        Image.fromarray(processed_img_data).save(output_path)

        return render_template('result.html', image_url=url_for('get_file', filename=output_filename), snr=snr, psnr=psnr)
    return "Invalid request", 400

@app.route('/compress')
def compress():
    return render_template('image_compression.html')

@app.route('/sharpen')
def sharpen():
    return render_template('image_sharpening.html')

@app.route('/grayscale')
def grayscale():
    return render_template('image_grayscale.html')

@app.route('/reduce_noise')
def reduce_noise():
    return render_template('image_noise_reduction.html')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
