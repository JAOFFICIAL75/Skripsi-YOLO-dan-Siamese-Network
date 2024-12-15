import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
import cv2
import random
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# https://www.semanticscholar.org/paper/Using-Siamese-Networks-with-Transfer-Learning-for-Heidari-Fouladi-Ghaleh/cfb95336ef93b2ee866e0c74daab647114c1067d
# https://stackoverflow-com.translate.goog/questions/28232235/how-to-calculate-the-number-of-parameters-of-convolutional-neural-networks?_x_tr_sl=en&_x_tr_tl=id&_x_tr_hl=id&_x_tr_pto=tc
# https://github.com/Chirag-Shilwant/One-Shot-Classification-using-Siamese-Network-on-MNIST-Dataset/blob/main/Siamese_Network_on_MNIST.ipynb
# https://medium.com/data-science-in-your-pocket/understanding-siamese-network-with-example-and-codes-e7518fe02612

result_folder = "Hasil Training Model\\databaru_sendiri_new"

try:
    os.mkdir(result_folder)
except:
    pass


def create_model():
    inputs = Input((160, 96, 3))
    x = Conv2D(64, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    flatten = Flatten()(x)
    fcl = Dense(128, activation="relu")(flatten)
    fcl = Dense(64, activation="relu")(fcl)

    model = Model(inputs, fcl)
    return model

feature_extractor = create_model()
print(feature_extractor.summary())
print()
left_input = Input((160,96,3))
right_input = Input((160,96,3))
featA = feature_extractor(left_input)
featB = feature_extractor(right_input)



def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = K.sum(K.square(featA - featB), axis=1, keepdims=True)
    return K.sqrt(sum_squared)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred, margin=1.0):
    # y_true contains binary labels (0 for similar, 1 for dissimilar)
    y_true = tf.cast(y_true, tf.float32)
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


distance = Lambda(euclidean_distance)([featA, featB])
siamese_net = Model(inputs=[left_input, right_input], outputs=distance)

optimizer = Adam()


#siamese_net.load_weights(r"C:\Users\Asus\.atom\databaru\SiamesePerson_CNN_Market1501_shuffle.h5", by_name=True)
siamese_net.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])
print(siamese_net.summary())
input("....")
model_json = siamese_net.to_json()
with open(f"{result_folder}\\siamese_model.json", "w") as json_file:
    json_file.write(model_json)



# Load Data
path = "hasil"
target_size = (96, 160)

image_list = []
label_list = []

folders = os.listdir(path)

images_dict = {folder: os.listdir(os.path.join(path, folder)) for folder in folders}

#for folder in images_dict:
#    images_dict[folder] = images_dict[folder][:20]

while any(images_dict.values()):
    for folder in folders:
        if images_dict[folder]:
            image = images_dict[folder].pop(0)
            img = cv2.imread(os.path.join(path, folder, image))
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            image_list.append(img)
            label_list.append(folder)
print(len(label_list))
print("")

left_input = []
right_input = []
targets = []

val_left = []
val_right = []
val_targets = []


#Number of pairs per image
pairs = 19
count = 0

for i in tqdm(range(len(label_list))):
    if count < int((pairs * 80)//100):
        data_comparing = []
        label_comparing = []
        for _ in range(pairs//2):
            while True: #Make sure it's not comparing to itself
                compare_to = random.randint(0, len(label_list)-1)
                if not compare_to in data_comparing:
                    if label_list[i] != label_list[compare_to] and compare_to != i:
                        if label_list[compare_to] not in label_comparing:
                            data_comparing.append(compare_to)
                            label_comparing.append(label_list[compare_to])
                            break
            left_input.append(image_list[i])
            right_input.append(image_list[compare_to])
            if label_list[i] != label_list[compare_to]:
                targets.append(0)
        data_comparing = []
        for _ in range(pairs//2):
            while True: #Make sure it's not comparing to itself
                compare_to = random.randint(0, len(label_list)-1)
                if not compare_to in data_comparing:
                    if label_list[i] == label_list[compare_to] and compare_to != i:
                        data_comparing.append(compare_to)
                        break
            left_input.append(image_list[i])
            right_input.append(image_list[compare_to])
            if label_list[i] == label_list[compare_to]:
                targets.append(1)
    else:
        data_comparing = []
        label_comparing = []
        for _ in range(pairs//2):
            while True: #Make sure it's not comparing to itself
                compare_to = random.randint(0, len(label_list)-1)
                if not compare_to in data_comparing:
                    if label_list[i] != label_list[compare_to] and compare_to != i:
                        if label_list[compare_to] not in label_comparing:
                            data_comparing.append(compare_to)
                            label_comparing.append(label_list[compare_to])
                            break
            val_left.append(image_list[i])
            val_right.append(image_list[compare_to])
            if label_list[i] != label_list[compare_to]:
                val_targets.append(0)
        data_comparing = []
        for _ in range(pairs//2):
            while True: #Make sure it's not comparing to itself
                compare_to = random.randint(0, len(label_list)-1)
                if not compare_to in data_comparing:
                    if label_list[i] == label_list[compare_to] and compare_to != i:
                        data_comparing.append(compare_to)
                        break
            val_left.append(image_list[i])
            val_right.append(image_list[compare_to])
            if label_list[i] == label_list[compare_to]:
                val_targets.append(1)
    count += 1
    if count == pairs:
        count = 0


left_input = np.squeeze(np.array(left_input))
right_input = np.squeeze(np.array(right_input))
targets = np.squeeze(np.array(targets))


val_left = np.squeeze(np.array(val_left))
val_right = np.squeeze(np.array(val_right))
val_targets = np.squeeze(np.array(val_targets))

print(len(left_input), len(right_input), len(targets))
print(len(val_left), len(val_right), len(val_targets))



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=result_folder+"\\SiamesePerson_CNN_Sendiri_contrastiveloss_{epoch:02d}.h5",
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_weights_only=True,
    save_best_only=False)
model_fit = siamese_net.fit([left_input,right_input], targets,
          batch_size=8,
          epochs=5,
          verbose=1,
          validation_data=([val_left,val_right],val_targets),
          callbacks=[model_checkpoint_callback])

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam || Loss : Contrastive Loss', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(model_fit.history['loss'], label='Training Loss')
plt.plot(model_fit.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(model_fit.history['accuracy'], label='Training Accuracy')
plt.plot(model_fit.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.savefig(f'{result_folder}\\plot.png')



train_loss = model_fit.history['loss']
val_loss = model_fit.history['val_loss']
train_accuracy = model_fit.history['accuracy']
val_accuracy = model_fit.history['val_accuracy']

history_data = np.array([train_loss, val_loss, train_accuracy, val_accuracy])

file_path = f'{result_folder}\\training_history.txt'
np.savetxt(file_path, history_data, delimiter=',')


# NOTE:
# Jika menggunakan model ini maka hasil sama akan mendekati nol
