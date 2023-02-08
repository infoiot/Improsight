import cv2
import numpy as np
import threading
import os
import datetime
file3="C:\\Users\\Dell\\Documents\\infoa_python\\Dockertest\\Imp_yolov4.cfg"
file4="E:\\Python\\Imp_yolov4_best.weights"
for folder, subs, files in os.walk("C:\\Users\\Dell\\Documents\\infoa_python\\Dockertest\\IMG\\IMPROSIGHT\\"):
    file_paths = []
    for filename in files:
        file_paths.append(os.path.abspath(os.path.join(folder, filename)))
        print("File list",file_paths)
        length=len(file_paths)
        print("Length",length)
        if length > 0 :
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
                        
                        #yolo_layers = [layers[i - 1] for i in network.getUnconnectedOutLayers()]
                        #yolo_model = cv2.dnn.readNetFromDarknet('F:\\Simulator_delivery\\Improsight_4.0\\EncryptionTest\\Imp_yolov4.cfg','F:\\Simulator_delivery\\Improsight_4.0\\EncryptionTest\\Imp_yolov4_best.weights')
                        yolo_layers = yolo_model.getLayerNames()
                        #yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
                        yolo_output_layer = [yolo_layers[i - 1] for i in yolo_model.getUnconnectedOutLayers()]
                        yolo_model.setInput(img_blob)
                        obj_detection_layers = yolo_model.forward(yolo_output_layer)
                        #%%############# NMS Change 1 ###############
                        # initialization for non-max suppression (NMS)
                        # declare list for [class id], [box center, width & height[], [confidences]
                        class_ids_list = []
                        boxes_list = []
                        confidences_list = []
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
                            max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
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
                        cv2.imwrite("C:\\Users\\Dell\\Documents\\infoa_python\\Dockertest\\IMG\\IMPROSIGHT\\Analysed.jpg", img)
                        print("Particle Count",pc)
                        print("Fibre Count",fc)
                        print("Gel Count",gc)
                        print("Air Bubble Count",abc)
                        file = open('C:\\Users\\Dell\\Documents\\infoa_python\\Dockertest\\IMG\\IMPROSIGHT\\rpt.txt','w')
                        file.write(str(pc)+"#"+str(fc)+"#"+str(gc)+"#"+str(abc))
                        file.close()
                        #os.remove(fn)
                        #os.remove(file3)
                        #os.remove(file4)
                        file_22 = open("C:\\Users\\Dell\\Documents\\infoa_python\\Dockertest\\res.txt", "w") 
                        #file_22 = open("res.txt", "w") 
                        file_22.write("Success")
                        file_22.close()
                    except Exception as e:
                        print(e)
                else:
                    print('File Content not proper')
        else:
            print('No File Found')
                        
