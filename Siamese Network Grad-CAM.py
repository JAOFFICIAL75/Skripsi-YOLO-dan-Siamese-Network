from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
# https://tree.rocks/get-heatmap-from-cnn-convolution-neural-network-aka-grad-cam-222e08f57a34


# Define the Siamese network architecture
def create_siamese_model():
    inputs = Input((160, 96, 3))
    x = Conv2D(64, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    flatten = Flatten()(x)
    fcl = Dense(128, activation="relu")(flatten)
    fcl = Dense(64, activation="relu")(fcl)

    model = Model(inputs=inputs, outputs=fcl)
    return model

def create_inference_model(siamese_model):
    left_input = Input((160, 96, 3))
    right_input = Input((160, 96, 3))

    featA = siamese_model(left_input)
    featB = siamese_model(right_input)

    def euclidean_distance(vectors):
        (featA, featB) = vectors
        sum_squared = K.sum(K.square(featA - featB), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([featA, featB])
    return Model(inputs=[left_input, right_input], outputs=distance)

# Load the saved weights
siamese_model = create_siamese_model()
inference_model = create_inference_model(siamese_model)
inference_model.load_weights(r"C:\Users\Asus\.atom\Siamese Network Tracking\Hasil Training Model\databaru_sendiri_new\SiamesePerson_CNN_Sendiri_contrastiveloss_03.h5")


# Load and preprocess an input image
#img_path = r'C:\Users\Asus\.atom\Siamese Network Tracking\Hasil Tracking\Hasil Realtime_Final\Hasil Data Realtime\Zidan\zidan_3177_0.429816.jpg'
img_path = r'C:\Users\Asus\.atom\Siamese Network Tracking\Hasil Tracking\Hasil Realtime_Final\Hasil Data Realtime\Zidan.jpg'
#img_path = r'C:\Users\Asus\.atom\Siamese Network Tracking\Hasil Tracking\Hasil Realtime_Final\Hasil Data Realtime\Zidan\zidan_2194_0.558163.jpg'
#img_path = r'C:\Users\Asus\.atom\Siamese Network Tracking\Hasil Tracking\Hasil Realtime_Final\Hasil Data Realtime\Zidan\zidan_4854_0.379590.jpg'
#img_path = r'C:\Users\Asus\.atom\Siamese Network Tracking\Hasil Tracking\Hasil Testing_Final\hasil tanpa threshold\testing1\track\Dimas.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (96, 160))
img = np.expand_dims(img, axis=0) / 255.0  # Normalize input image



conv_output = siamese_model.get_layer("max_pooling2d_1").output
pred_output = siamese_model.get_layer("dense_1").output
model = Model(siamese_model.input, outputs=[conv_output, pred_output])
conv, pred = model.predict(img)

conv_height, conv_width, _ = conv[0].shape  # Get the size of the activation map
scale_height = img.shape[1] / conv_height   # Scale factor for height
scale_width = img.shape[2] / conv_width     # Scale factor for width

# Plot the resized activation maps
#plt.figure(figsize=(16, 16))
#for i in range(36):
#    plt.subplot(6, 6, i + 1)
#    plt.imshow(img[0])  # Display the original image
#    resized_activation = zoom(conv[0, :, :, i], (scale_height, scale_width))  # Resize activation map
#    plt.imshow(resized_activation, cmap='jet', alpha=0.3)  # Overlay the activation map
#    plt.axis('off')
#plt.show()


# Compute the Grad-CAM heatmap
target = np.argmax(pred, axis=1).squeeze()
w, b = siamese_model.get_layer("dense_1").get_weights()
weights = w[:, target]  # Get the weights for the specific target class
heatmap = np.dot(conv[0], weights)  # Calculate the heatmap

# Show the Grad-CAM heatmap
heatmap_resized = zoom(heatmap, (scale_height, scale_width))  # Resize heatmap to input image size
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img[0])
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)
#plt.imshow(img[0])
#plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
#plt.title('Grad-CAM Heatmap without Image')
plt.colorbar()  # Add color bar here
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(img[0])
plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
plt.title('Grad-CAM Heatmap with Image')
plt.axis('off')
plt.show()
