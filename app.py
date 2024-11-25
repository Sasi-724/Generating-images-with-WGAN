from flask import Flask, render_template, jsonify
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Assuming you've already loaded your WGAN generator model
generator = tf.keras.models.load_model('WGAN_generator.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate_image():
    latent_dim = 100 
    # Generate a random noise input for the generator
    noise = tf.random.normal([1, latent_dim])  # Ensure latent_dim is defined
    generated_image = generator(noise, training=False)
    
    # Convert the generated image to a format that can be displayed in HTML
    image_array = generated_image.numpy().squeeze()  # Remove batch dimension
    image = Image.fromarray(((image_array + 1) * 127.5).astype(np.uint8))  # Rescale back to [0, 255]
    
    # Save image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({'image': img_str})

if __name__ == "__main__":
    app.run(debug=True)
