

import os
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time

modelfolder = 'yolov5models/'
yolov5allmodels=["yolov5n-diorall-int8",
"yolov5n-diorall-fp16",
"yolov5s-diorall-int8",
"yolov5s-diorall-fp16"]

modellisttpu=["yolov5s-diorall-int8_edgetpu",
"yolov5n-diorship-int8_edgetpu",
"yolov5s-diorship-int8_edgetpu",
"yolov5n-diorall-int8_edgetpu"]

path = "/home/odroid/work/testing/yolov5-export-to-raspberry-pi/dior-all-images"
directories = [d for d in os.listdir(path)]
runs = 100

def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
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
        #img_to_tensor =np.around(img_to_tensor)
        img_to_tensor = img_to_tensor.astype(input_type)
    return img_to_tensor





    







def processall(model, imgsize):
    start_time = time.perf_counter()
    start_cpu_time = time.process_time()
    #interpreter = tflite.Interpreter(model_path=modelfolder+model+'.tflite',num_threads=4)
    interpreter = tflite.Interpreter(model_path=modelfolder+model+'.tflite',num_threads=4,experimental_delegates=[tflite.load_delegate(library="/home/odroid/work/ArmNN-aarch64/libarmnnDelegate.so",options={"backends": "GpuAcc, CpuAcc,CpuRef"})])
    #interpreter = tflite.Interpreter(model_path="yolov5models/"+model+'.tflite',num_threads=4,experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()



    
    end_time = time.perf_counter()
    end_cpu_time = time.process_time()
            

    inference_time = end_time - start_time
    cpu_time = end_cpu_time - start_cpu_time
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    
    with open('logfiletime/'+model+'.txt', 'a') as f:
         f.write("init "+f"{cpu_time:.6f} {inference_time:.6f}\n")


    invoketimerecordings=[]

        
    for i in range(runs):
    
        start_time = time.perf_counter() 
        start_cpu_time = time.process_time()
        img = cv2.imread('dior-all-images/'+directories[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if imgsize == 448 or imgsize== 256:
           img = cv2.resize(img, (imgsize,imgsize))
        img_to_tensor = loadimg(img,input_details)
        
        
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
                

        inference_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        

        with open('logfiletime/'+model+'.txt', 'a') as f:
             f.write("loadimgtotensor "+f"{cpu_time:.6f} {inference_time:.6f}\n")
        

        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
            
        interpreter.set_tensor(input_details[0]['index'], img_to_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])



        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
                

        inference_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        

        with open('logfiletime/'+model+'.txt', 'a') as f:
             f.write("inference "+f"{cpu_time:.6f} {inference_time:.6f}\n")
        
        
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        output_type = output_details[0]['dtype']
        if output_type == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

        
        xyxy, classes, scores = YOLOdetect(output_data) 
        
        boxes = []
        H, W = img.shape[:2]
        
        for j in range(len(scores)):
            xmin = int(xyxy[0][j]*W)
            ymin = int(xyxy[1][j]*H)
            xmax = int(xyxy[2][j]*W)
            ymax = int(xyxy[3][j]*H)
            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.1, 0.7)
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
            
        inference_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        
        with open('logfiletime/'+model+'.txt', 'a') as f:
             f.write("post-processing-NMSBOXES "+f"{cpu_time:.6f} {inference_time:.6f}\n")
        
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        
        for idx in indices:
            xmin = boxes[idx][0]
            ymin = boxes[idx][1]
            xmax = boxes[idx][2]+boxes[idx][0]
            ymax = boxes[idx][3]+boxes[idx][1]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output/'+directories[i], img)
        
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
            
        inference_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        
        with open('logfiletime/'+model+'.txt', 'a') as f:
             f.write("post-processing-imgandboxes "+f"{cpu_time:.6f} {inference_time:.6f}\n")
             
             
yolov5comparison=["yolov5n-diorall-int8-256",
"yolov5n-diorall-fp16-256"]



if __name__ == '__main__':
    for model in yolov5shipmodels:
        for i in range(5):
            processall(model, 0)
#        j=j+1

    for model in yolov5comparison:
        for i in range(5):
            processall(model, 256)

