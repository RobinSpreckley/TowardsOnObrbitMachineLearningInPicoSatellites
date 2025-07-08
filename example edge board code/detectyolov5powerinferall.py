

def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):        
        classes.append(classdata[i].argmax())
    return classes 

def YOLOdetect(output_data): 
    output_data = output_data[0]       
    boxes = np.squeeze(output_data[..., :4])  
    scores = np.squeeze(output_data[..., 4:5]) 
    classes = classFilter(output_data[..., 5:])
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] 
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2] 

    return xyxy, classes, scores  




def loadimg(img,input_details):
    img_to_tensor= np.array(np.expand_dims(img,0), dtype=np.float32)/255
    input_type = input_details[0]['dtype']
    if input_type == np.uint8 or input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        img_to_tensor = (img_to_tensor / input_scale) + input_zero_point
        img_to_tensor = img_to_tensor.astype(input_type)
    return img_to_tensor







    












import socket
receiver_ip = '192.168.0.143'
receiver_port = 5000
import sys
import os
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
runs = 100
path = "/home/odroid/work/testing/yolov5-export-to-raspberry-pi/dior-all-images"
model = sys.argv[1]
imgsize = int(sys.argv[2])
directories = [d for d in os.listdir(path)]

def processall():
    #interpreter = tflite.Interpreter(model_path=model+'.tflite',num_threads=4)
    interpreter = tflite.Interpreter(model_path=model+'.tflite',num_threads=4,experimental_delegates=[tflite.load_delegate(library="/home/odroid/work/ArmNN-aarch64/libarmnnDelegate.so", options={"backends": "GpuAcc,CpuAcc,CpuRef"})])
    #interpreter = tflite.Interpreter(model_path=model+'.tflite',num_threads=4,experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    
    for i in range(runs):
        img = cv2.imread('dior-all-images/'+directories[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if imgsize == 448 or imgsize== 256:
           img = cv2.resize(img, (imgsize,imgsize))
        img_to_tensor = loadimg(img,input_details)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.connect((receiver_ip, receiver_port))
        message = 'start '+model+'infer'
        sock.sendall(message.encode())
        sock.close()

            
        interpreter.set_tensor(input_details[0]['index'], img_to_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.connect((receiver_ip, receiver_port))
        message = 'stop'
        msg = message.encode()
        sock.sendall(message.encode())
        sock.close()
        
        output_type = output_details[0]['dtype']
        if output_type == np.uint8 or output_type == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

        xyxy, classes, scores = YOLOdetect(output_data) 
        boxes = []
        drawboxes = []
        H, W = img.shape[:2]
        
        for j in range(len(scores)):
            xmin = int(xyxy[0][j]*W)
            ymin = int(xyxy[1][j]*H)
            xmax = int(xyxy[2][j]*W)
            ymax = int(xyxy[3][j]*H)
            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.1, 0.7)
        # Draw the boxes with NMS applied
        for idx in indices:
            xmin = boxes[idx][0]
            ymin = boxes[idx][1]
            xmax = boxes[idx][2]+boxes[idx][0]
            ymax = boxes[idx][3]+boxes[idx][1]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            #cv2.putText(img, str(classes[idx]), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.putText(img, str(scores[idx])[:4], (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output/'+directories[i], img)
        
        
processall()

