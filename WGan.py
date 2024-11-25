#!/usr/bin/env python
# coding: utf-8

# #### Contribution: 
# 
# * Karun Mehta - Loaded the data, did Eploratory Data Analysis & preprocessing required.  
# * Pavan Kalyan Meda - created Generator and Discriminator for WGAN along with WGAN Class. trained on the preprocessed images, generated few images. 
# * Yamini guduru - Deployed the saved models using Flask and created Demo video. calculated Inception, FID scores.
# 

# # Generated Images
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)
# ![image-4.png](attachment:image-4.png)
# ![image-5.png](attachment:image-5.png)

# #### Scores
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[1]:


#Loading the basic modules. 
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

#For images handling.
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#model & layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras.preprocessing import image


# In[2]:


df = pd.read_json('./yelp_photos/photos.json', lines=True)
df.head()


# In[26]:


df.info()


# In[27]:


df['label'].value_counts()


# In[3]:


df['photo_id'] = df['photo_id']+'.jpg'
df['image_path'] = './yelp_photos/photos/'+df['photo_id']


# In[4]:


import os
files_found = os.listdir('./yelp_photos/photos')
print("Photos found are:", files_found)


# In[5]:


filtered_df = df[df['photo_id'].isin(files_found)]


# In[28]:


filtered_df.info()


# In[29]:


filtered_df['label'].value_counts()


# In[6]:


import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image

def display_image(image_path):
    img = mpimg.imread(image_path)  # Read the image
    plt.figure(figsize=(6,6))  # Optional: adjust the size of the image
    plt.imshow(img)  # Display the image
    plt.axis('off')  # Remove axes
    plt.show()  # Show the imagenum_classes = len(categories)


# In[7]:


labels = filtered_df['label'].unique()

for label in labels:
    label_data = filtered_df[filtered_df['label'] == label].sample(n=1)
    image_path = label_data['image_path'].values[0]
    print(f"Displaying an image from the label: '{label}' ")
    display_image(image_path)


#  

# In[8]:


# Function to load and preprocess images
def image_preprocessing( labels, img_size = 64, label_sample_size = 150):
    X_train = []
    y_train = []
    for idx, label in enumerate(labels):
        sample_df = filtered_df[filtered_df['label'] == label].sample(n=label_sample_size)
        for image_path in sample_df['image_path']:
            img = load_img(image_path, target_size=(img_size, img_size))
            img_array = img_to_array(img)
            img_array = (img_array - 127.5) / 127.5
            X_train.append(img_array)
            y_train.append(idx)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


# In[9]:


# Load and preprocess images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
X_train, y_train = image_preprocessing(labels)


# In[10]:


print(X_train.shape, y_train.shape)


# ### WGAN Model

# In[12]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import numpy as np


# In[11]:


train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=1024).batch(32)


# In[13]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU

# Generator Network - Modified Version
def create_generator(latent_dim):
    generator = tf.keras.Sequential([
        Dense(8 * 8 * 128, input_dim=latent_dim, use_bias=False),
        Reshape((8, 8, 128)),
        Conv2DTranspose(128, kernel_size=(8, 8), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv2DTranspose(64, kernel_size=(8, 8), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv2DTranspose(3, kernel_size=(8, 8), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return generator

# Discriminator Network - Modified Version
def create_discriminator():
    discriminator = tf.keras.Sequential([
        Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=(64, 64, 3)),
        LeakyReLU(0.2),
        Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return discriminator


# In[14]:


# Define the WGAN model
class WGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(WassersteinGAN, self).__init__()
        self.generator_model = generator
        self.discriminator_model = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer):
        super(WassersteinGAN, self).compile()
        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # Train the discriminator
        for _ in range(5):
            random_noise = tf.random.normal([batch_size, latent_dim])
            generated_images = self.generator_model(random_noise)

            with tf.GradientTape() as disc_tape:
                real_preds = self.discriminator_model(real_images, training=True)
                fake_preds = self.discriminator_model(generated_images, training=True)

                disc_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds)

            disc_gradients = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator_model.trainable_variables))

            # Clip discriminator weights (WGAN weight clipping)
            for param in self.discriminator_model.trainable_variables:
                param.assign(tf.clip_by_value(param, -0.01, 0.01))

        # Train the generator
        random_noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator_model(random_noise)
            fake_preds = self.discriminator_model(generated_images, training=True)

            gen_loss = -tf.reduce_mean(fake_preds)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator_model.trainable_variables))


        return {'d_loss': d_loss, 'g_loss': g_loss}


# In[15]:


# Parameters
latent_dim = 100
g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

# Build models
generator = create_generator(latent_dim)
discriminator = create_discriminator()

# Create WGAN model
wgan = WGAN(generator, discriminator)
wgan.compile(g_optimizer, d_optimizer)


# In[16]:


# Train the WGAN
wgan.fit(train_dataset, epochs=1000, verbose=1)


# In[17]:


# Generate new images
latent_dim = 100  # Dimensionality of the generator input
num_images_to_generate = 5  # Number of images to generate

# Generate random noise
random_noise = np.random.normal(size=(num_images_to_generate, latent_dim))

# Use the trained generator to generate images
generated_images = generator.predict(random_noise)

# Display or save generated images
for i in range(num_images_to_generate):
    plt.imshow((generated_images[i] + 1) / 2)  # Scale images from [-1, 1] to [0, 1] for display
    plt.axis('off')
    plt.show()


# In[18]:


import tensorflow as tf
import numpy as np

def compute_inception_score(images, batch_size=32):
    # Initialize the InceptionV3 model for feature extraction
    inception_net = tf.keras.applications.InceptionV3(
        include_top=False, pooling='avg', input_shape=(299, 299, 3)
    )
    
    # Resize images to the input size expected by InceptionV3
    resized_images = tf.image.resize(images, (299, 299))
    
    # Obtain feature representations using the InceptionV3 model
    feature_map = inception_net.predict(resized_images, batch_size=batch_size)
    
    # Calculate the Inception Score (IS)
    kl_divergence = np.sum(feature_map * np.log(feature_map + 1e-6), axis=1)
    inception_score = np.exp(np.mean(kl_divergence))
    
    return inception_score


# In[19]:


import tensorflow as tf
import numpy as np
from scipy import linalg

def compute_frechet_distance(real_images, generated_images, batch_size=32):
    # Load InceptionV3 model for feature extraction
    inception_v3_model = tf.keras.applications.InceptionV3(
        include_top=False, pooling='avg', input_shape=(299, 299, 3)
    )
    
    # Resize images to fit the input dimensions of InceptionV3
    real_resized = tf.image.resize(real_images, (299, 299))
    generated_resized = tf.image.resize(generated_images, (299, 299))
    
    # Extract features for both real and generated images
    real_features = inception_v3_model.predict(real_resized, batch_size=batch_size)
    generated_features = inception_v3_model.predict(generated_resized, batch_size=batch_size)
    
    # Compute the mean and covariance of the features for both real and generated images
    mean_real, cov_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mean_generated, cov_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    
    # Compute the difference between the means
    mean_diff = mean_real - mean_generated
    
    # Calculate the square root of the product of the covariance matrices
    cov_sqrt, _ = linalg.sqrtm(cov_real.dot(cov_generated), disp=False)
    
    # Return the Frechet Inception Distance (FID)
    fid = np.sum(mean_diff**2) + np.trace(cov_real + cov_generated - 2 * cov_sqrt)
    
    return fid


# In[20]:


is_score = calculate_inception_score(generated_images)


# In[21]:


print(f"Inception Score: {is_score}")


# In[22]:


from scipy import linalg
val_images,val_labels = image_preprocessing(labels, label_sample_size=10)
fid_score = calculate_frechet_distance(val_images, generated_images)


# In[23]:


print(f"Frechet Inception Distance: {fid_score}")


# In[25]:


generator.save('WGAN_generator.h5')
discriminator.save('WGAN_discriminator.h5')


# In[ ]:




