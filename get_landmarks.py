
# Detect faces and then draw face landmarks directly on the faces
# Dataset: http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html (BadDataset)
# Dataset2: http://mrl.cs.vsb.cz/eyedataset
######################################################################################

import cv2
import numpy as np
import dlib
from imutils import face_utils
import sys # for command line arguments
from time import sleep
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

######################################################################################
# NEXT STEPS:                                                                        #
#               - Smooth cropped eyes                                                #
#               - Try using edges (e.g. morphological operator GRADIENT)             #
######################################################################################

# utility function used to transform an image to a tensor in order to be used as input to the CNN
def prepare_image(img):
    img_t = np.array(img)                   # save image in np array
    # transform the np array in a tensor
    img_t = np.expand_dims(img_t, axis=0)   # add axis
    img_t = np.expand_dims(img_t, axis=3)   # add axis

    return img_t

# Use webcam
def get_video_landmarks(detector, predictor, model):
    
    classes = ['Closed', 'Open']
    img_shape = 112 # dimension used in the CNN
    #################
    #################

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            shape = face_utils.shape_to_np(landmarks)


            #################################
            # SHOULD CREATE A FUNCTION HERE #
            #################################

            # important points for eyes detection
            nose_point = shape[29][1]

            # Right eye
            right_eye_left_x = shape[17][0]
            right_eye_right_x = shape[21][0]
            right_eye_top = shape[19][1]
            right_eye = frame[right_eye_top:nose_point, right_eye_left_x:right_eye_right_x]
            right_eye = cv2.resize(right_eye, (img_shape,img_shape), cv2.INTER_CUBIC)
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            right_eye = cv2.equalizeHist(right_eye)
            # right_eye = cv2.GaussianBlur(right_eye,(3,3), 0)

            # Left eye
            left_eye_left_x = shape[22][0]
            left_eye_right_x = shape[26][0]
            left_eye_top = shape[24][1]
            left_eye = frame[left_eye_top:nose_point, left_eye_left_x:left_eye_right_x]
            left_eye = cv2.resize(left_eye, (img_shape,img_shape), cv2.INTER_CUBIC)
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            left_eye = cv2.equalizeHist(left_eye)
            
            # Draw on original image
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


            cv2.imshow('Left', left_eye)
            cv2.imshow('Right', right_eye)
            pred = model.predict_classes(prepare_image(left_eye))
            left_pred = classes[pred[0]]
            print("Left eye is: " + classes[pred[0]])
            pred = model.predict_classes(prepare_image(right_eye))
            right_pred = classes[pred[0]]
            print("Right eye is: " + classes[pred[0]])

            if right_pred == 'Closed' and left_pred == 'Closed':
                os.system('spd-say "both"')
            else:
                if right_pred == 'Closed':
                    os.system('spd-say "right"')  # speaker
                if left_pred == 'Closed':
                    os.system('spd-say "left"')  # speaker
            
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1)
        if key == 24:
            break
    cap.release()
    cv2.destroyAllWindows()

# Use images from the dataset
def get_image_landmarks(detector, predictor, model, path, waitkey):

    classes = ['Closed', 'Open']
    img_shape = 96 # dimension used in the CNN
    #################
    #################

    frame = cv2.imread(path)     
    # frame = cv2.resize(frame, (100,100))   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    preds = []
    for face in faces:
        # print("Detected a face")

        landmarks = predictor(gray, face)
        shape = face_utils.shape_to_np(landmarks)

        # important points for eyes detection
        nose_point = shape[29][1]

        # Right eye
        right_eye_left_x = shape[17][0]
        right_eye_right_x = shape[21][0]
        right_eye_top = shape[19][1]
        right_eye = frame[right_eye_top:nose_point, right_eye_left_x:right_eye_right_x]
        right_shape = right_eye.shape
        if (not right_shape[0] == 0) and (not right_shape[1] == 0):
            right_eye = cv2.resize(right_eye, (img_shape,img_shape), cv2.INTER_CUBIC)
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('Right', right_eye)
            pred = model.predict_classes(prepare_image(right_eye))
            # print("Right eye is: " + classes[pred[0]])
            # print(pred)
            if classes[pred[0]] == "Closed":
                preds.append(1)
            else:
                preds.append(0)

        # Left eye
        left_eye_left_x = shape[22][0]
        left_eye_right_x = shape[26][0]
        left_eye_top = shape[24][1]
        left_eye = frame[left_eye_top:nose_point, left_eye_left_x:left_eye_right_x]
        left_shape = left_eye.shape
        if (not left_shape[0] == 0) and (not left_shape[1] == 0):
            left_eye = cv2.resize(left_eye, (img_shape,img_shape), cv2.INTER_CUBIC)
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('Left', left_eye)
            pred = model.predict_classes(prepare_image(left_eye))
            # print("Left eye is: " + classes[pred[0]])
            # print(pred)
            if classes[pred[0]] == "Closed":
                preds.append(1)
            else:
                preds.append(0)       
            
        
        # Draw on original image
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # cv2.imshow('image', frame)
    # cv2.waitKey(waitkey)
    # cv2.destroyAllWindows()
    return preds

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

######################################################################################
#                               Load CNN Model                                       #
######################################################################################
json_file = open('NN/saved_models/naive_model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# summarize model.
# model.summary()
model.load_weights('NN/saved_models/naive_model2_weights_datagen.h5')
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# To run:
# $ python3 get_landmarks.py [video|path/to/image]

# let's use this for some tests:
if len(sys.argv) == 1:
    dir_path = "datasets/BadDataset/ClosedFace/"
    # dir_path = "datasets/BadDataset/OpenFace/"
    # dir_path = "datasets/mrlEyes/Close/"
    predicted = 0
    counter = 0
    errors = 0
    for i,filename in enumerate(os.listdir(dir_path)):
        # if i > 1000:
        #     break
        print(i)
        preds = get_image_landmarks(detector, predictor, model, dir_path+"/"+filename, 0)
        for p in preds:
            # print(p)
            if p == 0:      # 1 means Closed, 0 Open
                errors += 1
            predicted += p
            counter += 1
    predicted /= counter
    print("Predicted:", predicted)
    print("Error rate:", errors/counter)
elif sys.argv[1] == "video":
    get_video_landmarks(detector, predictor, model)
else:
    preds = get_image_landmarks(detector, predictor, model, sys.argv[1], 0)
    print(preds)