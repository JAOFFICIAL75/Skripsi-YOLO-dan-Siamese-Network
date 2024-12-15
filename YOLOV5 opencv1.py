# menggunakan tensorflow versi 2.10.0
# databaru_sendiri_v2 dan databaru_sendiri_v3 memberikan hasil yang stabil walau tidak selalu terdeteksi

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
import tkinter as tk
from tkinter import simpledialog
import random
import time
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cap = cv2.VideoCapture("Video Testing/testing1.mp4")
net = cv2.dnn.readNetFromONNX("dc 1.onnx")



# databaru_sendiri_v2
def create_siamese_model():
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



# Define the Siamese model for inference
def create_inference_model(siamese_model):
    left_input = Input((160,96,3))
    right_input = Input((160,96,3))
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

# Create the inference model
# bagus menggunakan v2 epoch 3 dan v3 epoch 1
inference_model = create_inference_model(siamese_model)
inference_model.load_weights("Hasil Training Model/databaru_sendiri_new/SiamesePerson_CNN_Sendiri_contrastiveloss_03.h5")


root = tk.Tk()
root.withdraw()

simpan_warna = {}
nama_track_tersimpan = {}
def add_new_object(frame):
    name = simpledialog.askstring("Enter Name", "Enter the name for the object:")
    if name is not None:
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object")  # Manually close the "Select Object" window
        if all(bbox):  # Check if the bounding box was selected (not canceled)
            x, y, w, h = bbox
            nama_track_tersimpan[name] = frame[y:y+h, x:x+w]
            simpan_warna[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #cv2.imwrite(f"{name}.jpg", frame[y:y+h, x:x+w])
    return


def preprocess_image(img, target_size):
    try:
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
    except Exception as e:
        print(e)
    return img

def compare_images(image_1, image_2):
    target_size = (96, 160)
    if len(image_1) != 0:
        img_1 = preprocess_image(image_1, target_size)
        img_2 = preprocess_image(image_2, target_size)
        similarity_score = inference_model.predict([img_1, img_2])[0][0]
        return similarity_score
    else:
        return 0


def draw_detection(img, r, color=(0, 255, 0), thickness=3):
    hor = r[2] // 7
    ver = r[3] // 7
    # Top left corner
    cv2.line(img, tuple(r[0:2]), (r[0], r[1] + ver), color, thickness)
    cv2.line(img, tuple(r[0:2]), (r[0] + hor, r[1]), color, thickness)
    # Top right corner
    cv2.line(img, (r[0] + r[2] - hor, r[1]), (r[0] + r[2], r[1]), color, thickness)
    cv2.line(img, (r[0] + r[2], r[1] + ver), (r[0] + r[2], r[1]), color, thickness)
    # Bottom right corner
    cv2.line(img, (r[0] + r[2], r[1] + r[3] - ver), (r[0] + r[2], r[1] + r[3]), color, thickness)
    cv2.line(img, (r[0] + r[2] - hor, r[1] + r[3]), (r[0] + r[2], r[1] + r[3]), color, thickness)
    # Bottom left corner
    cv2.line(img, (r[0], r[1] + r[3] - ver), (r[0], r[1] + r[3]), color, thickness)
    cv2.line(img, (r[0] + hor, r[1] + r[3]), (r[0], r[1] + r[3]), color, thickness)



def generate_contrast_color(bgr_color):
    bgr_color = np.array(bgr_color)
    contrast_color = 255 - bgr_color
    return (int(contrast_color[0]), int(contrast_color[1]), int(contrast_color[2]))

def draw_text(image, teks, x1, y1, color):
    text_width, text_height = cv2.getTextSize(text=teks, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                              thickness=2)[0]
    text_x = x1 + 5
    text_y = y1 - 5

    text_background_x1 = x1
    text_background_y1 = y1 - 2 * 5 - text_height

    text_background_x2 = x1 + 2 * 5 + text_width
    text_background_y2 = y1

    cv2.rectangle(image, pt1=(text_background_x1, text_background_y1), pt2=(text_background_x2, text_background_y2),
                  color=color, thickness=cv2.FILLED)
    contrast = generate_contrast_color(color)
    cv2.putText(image, text=teks, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=contrast, thickness=1, lineType=cv2.LINE_AA)


#frameSize = [1200, 640]
#cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out_path = 'Alena Bergerak_v2_update.mp4'
#out = cv2.VideoWriter(out_path,cv2_fourcc, cap.get(cv2.CAP_PROP_FPS), frameSize)
while True:
    img = cap.read()[1]
    if img is None:
        break
    img = cv2.resize(img, (1200, 640))
    image = img.copy()
    start_time = time.time()
    #pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]
    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes
    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.2:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.2:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.2,0.2)


    cek_kemiripan = {}
    for i in indices:
        x1,y1,w,h = boxes[i]
        #cv2.rectangle(img,(x1,y1),(x1+w,y1+h), (0, 255, 0),1)
        cv2.circle(img, (int(x1+(w/2)), int(y1+(h/2))), 3, (0, 255, 0), -1)
        person = image[y1:y1+h, x1:x1+w]
        for data in nama_track_tersimpan:
            if not data in cek_kemiripan:
                cek_kemiripan[data] = {"nilai" : [], "bbox" : []}
            nilai_kemiripan = compare_images(person, nama_track_tersimpan[data])
            cek_kemiripan[data]["nilai"].append(nilai_kemiripan)
            cek_kemiripan[data]["bbox"].append([x1, y1, w, h])
    for i in cek_kemiripan:
        if len( cek_kemiripan[i]["nilai"]) != 0:
            indeks = cek_kemiripan[i]["nilai"].index(min(cek_kemiripan[i]["nilai"]))
            nilai_minimal = min(cek_kemiripan[i]["nilai"])
            print(i, nilai_minimal)
            x1, y1, w, h = cek_kemiripan[i]["bbox"][indeks]
            #nama_track_tersimpan[i] = img[y1:y1+h, x1:x1+w]
            #if i == 'Azizah':
            #    cv2.imwrite(f"Hasil Data Realtime/Azizah/azizah_{int(pos_frame)}_{min(cek_kemiripan[i]['nilai']):.6f}.jpg", image[y1:y1+h, x1:x1+w])
            #if i == "Zidan":
            #    cv2.imwrite(f"Hasil Data Realtime/Zidan/zidan_{int(pos_frame)}_{min(cek_kemiripan[i]['nilai']):.6f}.jpg", image[y1:y1+h, x1:x1+w])
            #elif i == "Rizki":
            #    cv2.imwrite(f"Hasil Data Realtime/Rizki/rizki_{int(pos_frame)}_{min(cek_kemiripan[i]['nilai']):.6f}.jpg", image[y1:y1+h, x1:x1+w])
            if nilai_minimal <= 0.7:
                cv2.rectangle(img,(x1,y1),(x1+w,y1+h), simpan_warna[i],1)
                draw_detection(img, (x1, y1, w, h), color=simpan_warna[i], thickness=3)
                #cv2.putText(img, i, (x1,y1-4),cv2.FONT_HERSHEY_COMPLEX, 0.5,simpan_warna[i],2)
                draw_text(img, i, x1, y1, simpan_warna[i])
    end_time = time.time()
    fps = round((1/(end_time-start_time)), 2)
    print("FPS:", str(fps))


    #out.write(img)
    cv2.rectangle(img,(990,0), (1200,60), (0, 0, 0), -1)
    cv2.putText(img, f"FPS: {fps}", (1000, 40),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("VIDEO",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        add_new_object(image)
    if key == 27:
        break
#out.release()
