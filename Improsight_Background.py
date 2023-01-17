# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:13:43 2021

@author: user
"""
import cv2
import numpy as np
import threading
import os
import datetime
from cryptography.fernet import Fernet

file1="Imp_yolov4.cfg"
file2="Imp_yolov4_best.weights"
file3="C:\\Users\\Dell\\Documents\\infoa_python\\Dockertest\\Imp_yolov4.cfg"
file4="C:\\Users\\Dell\\Documents\\infoa_python\\Dockertest\\Imp_yolov4_best.weights"
#key=b'8cK5ftacBWd5JqU7b3jzS1BRs_20NDULinOtPWjfKvQ='
key=b'8cK5ftacBWd5JqU7b3jzS1BRs_20NDULinOtPWjfKvQ='
#b'8cK5ftacBWd5JqU7b3jzS1BRs_20NDULinOtPWjfKvQ='

def decrypt(filename,filename1,key):
    f=Fernet(key)
    #print(filename)
    #print(filename1)
    #print(key)
    with open(filename,"rb") as file:
        encrypted_data=file.read()
    decrypted_data=f.decrypt(encrypted_data)
    with open(filename1,"wb") as file:
        file.write(decrypted_data)
def load_key():
    return open("key.key","rb").read()


def analyse():
    file=open('path.txt',"r")
    fc=file.read()
    file.close()
    #print("Reading File Path",fc)
    fc1=fc.split("#")
    #%% Load image and saving in blob
    #print('key',key)
    if fc1[0] == 'VID':
        analyseVideoStream()
    elif fc1[0] == 'IMG':
        file_paths = []
        for folder, subs, files in os.walk(fc1[1]):
            for filename in files:
                file_paths.append(os.path.abspath(os.path.join(folder, filename)))
                print("File list",file_paths)
        length=len(file_paths)
        print("Length",length)
        if length > 0 :
            decrypt(file1,file3,key)
            decrypt(file2,file4,key)
            for fls in file_paths:
                fn=fls
                print("File Name with absolute path",fn)
                print('Is File',os.path.isfile(fn))
                if os.path.isfile(fn):
                    try:
                        img = cv2.imread(fn)
                        img_height = img.shape[0]
                        img_width = img.shape[1]
                        
                        img_blob = cv2.dnn.blobFromImage(img,0.003922,(416,416),swapRB = True,crop = False)
                        #%% Labeling class
                        class_labels = ["Gel","Particle","Fibre","Air Bubble"]
                        #%% color labeling
                        class_colors = ["0,255,255","0,0,255","255,0,0","255,255,0"]
                        class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
                        class_colors = np.array(class_colors)
                        class_colors = np.tile(class_colors,(4,1))
                        #%% Loading the model and forward the neural net
                        #decrypt(file,key)
                        #decrypt(file1,key)
                        yolo_model = cv2.dnn.readNetFromDarknet(file3,file4)
                        #yolo_model = cv2.dnn.readNetFromDarknet('F:\\Simulator_delivery\\Improsight_4.0\\EncryptionTest\\Imp_yolov4.cfg','F:\\Simulator_delivery\\Improsight_4.0\\EncryptionTest\\Imp_yolov4_best.weights')
                        yolo_layers = yolo_model.getLayerNames()
                        yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
                        yolo_model.setInput(img_blob)
                        obj_detection_layers = yolo_model.forward(yolo_output_layer)
                        #%%############# NMS Change 1 ###############
                        # initialization for non-max suppression (NMS)
                        # declare list for [class id], [box center, width & height[], [confidences]
                        class_ids_list = []
                        boxes_list = []
                        confidences_list = []
                        #%%############# NMS Change 1 END ###########
                        #%% Search for the object
                        for object_detection_layer in obj_detection_layers:
                            for object_detection in object_detection_layer:
                                all_scores = object_detection[5:]
                                predicted_class_id = np.argmax(all_scores)
                                prediction_confidence = all_scores[predicted_class_id]
                                if prediction_confidence > 0.5:
                                    predicted_class_label = class_labels[predicted_class_id]
                                    bounding_box = object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
                                    (cx,cy,w,h) = bounding_box.astype("int")
                                    x = int(cx -w/2)
                                    y = int(cy-h/2)
                                    ############## NMS Change 2 ###############
                                    #save class id, start x, y, width & height, confidences in a list for nms processing
                                    #make sure to pass confidence as float and width and height as integers
                                    class_ids_list.append(predicted_class_id)
                                    confidences_list.append(float(prediction_confidence))
                                    boxes_list.append([x, y, int(w), int(h)])
                                    ############## NMS Change 2 END ###########
                        ############## NMS Change 3 ###############
                        # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
                        # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
                        max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
                        # loop through the final set of detections remaining after NMS and draw bounding box and write text
                        fc=0
                        gc=0
                        pc=0
                        abc=0
                        for max_valueid in max_value_ids:
                            max_class_id = max_valueid[0]
                            box = boxes_list[max_class_id]
                            x = box[0]
                            y = box[1]
                            w = box[2]
                            h = box[3]
                            
                            #get the predicted class id and label
                            predicted_class_id = class_ids_list[max_class_id]
                            predicted_class_label = class_labels[predicted_class_id]
                            prediction_confidence = confidences_list[max_class_id]
                        ############## NMS Change 3 END ###########
                            wx = int(x+w)
                            hy = int(y+h)
                            box_color = class_colors[predicted_class_id]
                            box_color = [int(c) for c in box_color]
                            if predicted_class_label=="Fibre":
                                fc =fc+1
                            if predicted_class_label=="Gel":
                                gc=gc+1
                            if predicted_class_label=="Air Bubble":
                                abc=abc+1
                            if predicted_class_label=="Particle":
                                pc=pc+1
                                
                            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                            print("predicted object {}".format(predicted_class_label))           
                            # draw rectangle and text in the image
                            cv2.rectangle(img, (x, y), (wx, hy), box_color, 1)
                            cv2.putText(img, predicted_class_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                    
                        cv2.imwrite("Analysed.jpg", img)
                        print("Particle Count",pc)
                        print("Fibre Count",fc)
                        print("Gel Count",gc)
                        print("Air Bubble Count",abc)
                        file = open('rpt.txt','w')
                        file.write(str(pc)+"#"+str(fc)+"#"+str(gc)+"#"+str(abc))
                        file.close()
                        os.remove(fn)
                        os.remove(file3)
                        os.remove(file4)
                    except:
                        print('Exception in reading File')
                else:
                    print('File Content not proper')
        else:
            print('No File Found')
        
    return

def analyseVideoStream():
    file=open('path.txt',"r")
    fc=file.read()
    file.close()
    #print("Reading File Path",fc)
    fc1=fc.split("#")
    if fc1[0] == 'IMG':
        analyse()
    elif fc1[0] == 'VID':             
        file_paths = []
        for folder, subs, files in os.walk(fc1[1]):
            for filename in files:
                file_paths.append(os.path.abspath(os.path.join(folder, filename)))
                #print("File list",file_paths)
        length=len(file_paths)
        print("Length",length)
        if length > 0 :
            for fls in file_paths:
                fn=fls
                print("File Name with absolute path",fn)
                print('Is File',os.path.isfile(fn))
                if os.path.isfile(fn):
                    try:
                        #%% Capture the video
                        realVideoFrame = cv2.VideoCapture(fn)
                        while True:
                            decrypt(file1,file3,key)
                            decrypt(file2,file4,key)
                            ret,frame = realVideoFrame.read()
                            if not ret:
                                print("... end of video file reached");
                                break;
                            #%% Load image and saving in blob
                            #img = cv2.imread('images/testing/scene3.jpg')
                            img = frame
                            img_height = img.shape[0]
                            img_width = img.shape[1]
                            
                            img_blob = cv2.dnn.blobFromImage(img,0.003922,(416,416),swapRB = True,crop = False)
                            
                            #%% Labeling class
                            class_labels = ["Gel","Particle","Fibre","Air Bubble"]
                            #%% color labeling
                            class_colors = ["0,255,255","0,0,255","255,0,0","255,255,0"]
                            class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
                            class_colors = np.array(class_colors)
                            class_colors = np.tile(class_colors,(4,1))
                            #%% Loading the model and forward the neural net
                            yolo_model = cv2.dnn.readNetFromDarknet(file3,file4)
                            #yolo_model = cv2.dnn.readNetFromDarknet('Imp_yolov4.cfg','Imp_yolov4_best.weights')
                            yolo_layers = yolo_model.getLayerNames()
                            yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
                            yolo_model.setInput(img_blob)
                            obj_detection_layers = yolo_model.forward(yolo_output_layer)
                            
                            #%%############# NMS Change 1 ###############
                            # initialization for non-max suppression (NMS)
                            # declare list for [class id], [box center, width & height[], [confidences]
                            class_ids_list = []
                            boxes_list = []
                            confidences_list = []
                            #%%############# NMS Change 1 END ###########
                            
                            #%% Search for the object
                            for object_detection_layer in obj_detection_layers:
                                for object_detection in object_detection_layer:
                                    all_scores = object_detection[5:]
                                    predicted_class_id = np.argmax(all_scores)
                                    prediction_confidence = all_scores[predicted_class_id]
                                    if prediction_confidence > 0.5:
                                        predicted_class_label = class_labels[predicted_class_id]
                                        bounding_box = object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
                                        (cx,cy,w,h) = bounding_box.astype("int")
                                        x = int(cx -w/2)
                                        y = int(cy-h/2)
                                        ############## NMS Change 2 ###############
                                        #save class id, start x, y, width & height, confidences in a list for nms processing
                                        #make sure to pass confidence as float and width and height as integers
                                        class_ids_list.append(predicted_class_id)
                                        confidences_list.append(float(prediction_confidence))
                                        boxes_list.append([x, y, int(w), int(h)])
                                        ############## NMS Change 2 END ###########
                            ############## NMS Change 3 ###############
                            # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
                            # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
                            max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
                            fc=0
                            gc=0
                            pc=0
                            abc=0
                            # loop through the final set of detections remaining after NMS and draw bounding box and write text
                            for max_valueid in max_value_ids:
                                max_class_id = max_valueid[0]
                                box = boxes_list[max_class_id]
                                x = box[0]
                                y = box[1]
                                w = box[2]
                                h = box[3]
                                
                                #get the predicted class id and label
                                predicted_class_id = class_ids_list[max_class_id]
                                predicted_class_label = class_labels[predicted_class_id]
                                prediction_confidence = confidences_list[max_class_id]
                            ############## NMS Change 3 END ###########
                                wx = int(x+w)
                                hy = int(y+h)
                                box_color = class_colors[predicted_class_id]
                                box_color = [int(c) for c in box_color]
                                if predicted_class_label=="Fibre":
                                    fc =fc+1
                                if predicted_class_label=="Gel":
                                    gc=gc+1
                                if predicted_class_label=="Air Bubble":
                                    abc=abc+1
                                if predicted_class_label=="Particle":
                                    pc=pc+1
                                predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                                print("predicted object {}".format(predicted_class_label))           
                                # draw rectangle and text in the image
                                cv2.rectangle(img, (x, y), (wx, hy), box_color, 1)
                                cv2.putText(img, predicted_class_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                                cv2.imwrite("Analysed.jpg", img)
                                os.remove(file3)
                                os.remove(file4)
                                print("Particle Count",pc)
                                print("Fibre Count",fc)
                                print("Gel Count",gc)
                                print("Air Bubble Count",abc)
                                file = open('rpt.txt','w')
                                file.write(str(pc)+"#"+str(fc)+"#"+str(gc)+"#"+str(abc))
                                file.close()
                    except:
                        print('Exception in reading File')
            else:
                    print('File Content not proper')
        else:
            print('No File Found')

d1 = datetime.datetime(2023, 1, 10)
current_time = datetime.datetime.now()
year=current_time.year
mon=current_time.month
day=current_time.day
d2 = datetime.datetime(year,mon,day)
#key=load_key()
#print('key',key)
#decrypt(file,key)
#decrypt(file1,key)
if d1 >= d2 :
    file=open('path.txt',"r")
    fc=file.read()
    file.close()
    #print("Reading File Path",fc)
    fc1=fc.split("#")
    print('After Split',fc1)
    if fc1[0] == 'IMG':
        t=threading.Thread(target=analyse())
        t.start()
        while True:
            analyse()
    elif fc1[0] == 'VID':
        #print('Inside Video Analysis')
        t=threading.Thread(target=analyseVideoStream())
        t.start()
        while True:
            analyseVideoStream()
    elif fc1[0] == 'SCR':
        print('Inside Screen Capture')
else:
    print('Software License Expired')
