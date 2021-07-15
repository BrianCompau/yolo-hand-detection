import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import mediapipe as mp

W = 848
H = 480
fps = 30

from yolo import YOLO
obj_labels = open("models/yolov3-coco/coco-labels").read().strip().split('\n')
#yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
yolo.size = int(416)
yolo.confidence = float(0.2)
obj_model= YOLO("models/yolov3-coco/yolov3.cfg", "models/yolov3-coco/yolov3.weights", obj_labels)
obj_model.size = int(416)
obj_model.confidence = float(0.6)

rs_serial='031222070617'
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(rs_serial)


dec_filter = rs.decimation_filter()  # Decimation-reduce df density
spat_filter = rs.spatial_filter()    # Spatial-spatial smoothing
temp_filter = rs.temporal_filter()   # Temporal-reduces temporal noise

print("[INFO] start streaming...")
#profile = pipeline.start(config)
profile = pipeline.start(config)
pipeline.wait_for_frames()

import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import requests
import time

font = cv2.FONT_HERSHEY_SIMPLEX
API_ENDPOINT = 'https://t24blpcmsa.execute-api.us-east-1.amazonaws.com/dev'
fontScale = 1

color = (255, 0, 0)

thickness = 2
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
def getFingers(frame):
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        #image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detections
        #print(results)
        image_height, image_width, _ = image.shape


        # Rendering results
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                xIndexFinger =hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                yIndexFinger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y* image_height
                xMiddleFinger =hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
                yMiddleFinger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y* image_height
                cv2.putText(image, 'o', (int(xIndexFinger)-10,int(yIndexFinger)-10), font, fontScale, (255, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, 'o', (int(xMiddleFinger)-10,int(yMiddleFinger)-10), font, fontScale, (255, 255, 0), 4, cv2.LINE_AA)
                return image, xIndexFinger, yIndexFinger
        return image, False, False
        
        
def postCoords(coords, interactionDist):
    data = {"currentDist":coords,"interactDist":interactionDist}
  
    # sending post request and saving response as response object
    requests.post(url = API_ENDPOINT +'/setCoords', data = data)

# In[6]:



#device = profile.get_device()
#depth_sensor = device.first_depth_sensor()
#device.hardware_reset()

#points = point_cloud.calculate(depth_frame)
#color_frame = frameset_before.get_color_frame()
    
#verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz
# Convert images to numpy arrays

    

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)
point_cloud = rs.pointcloud()
hands = 1

def getIntelFrames():
    
    frames = pipeline.wait_for_frames()
    #frames = dec_filter.process(frames).as_frameset()
    #frames = spat_filter.process(frames).as_frameset()
    #frames = temp_filter.process(frames).as_frameset()
    aligned_frames = align.process(frames)  
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    
    return color_image,depth

    

def showColorizedDepth(depth_frame, detection,distanceString):
    id, name, confidence, x, y, w, h = detected_instance
    color = (0, 255, 255)
   
    xmin_depth = x+round(w/4)
    ymin_depth = y+round(h/4)
    xmax_depth = x+round(w*.75)
    ymax_depth = y+round(h*.75) 
    
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    cv2.rectangle(colorized_depth, (xmin_depth, ymin_depth), 
                 (xmax_depth, ymax_depth), (255, 255, 255), 2)

    cv2.putText(colorized_depth, distanceString, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    cv2.imshow("Depth Frame",colorized_depth)

        
cell_phone_flag= False
set_interaction=False


def cellPhoneDetection(frame, depth):
    width_obj, height_obj, inference_time_obj, results_obj = obj_model.inference(frame)
    for detection in results_obj:
        id, name, confidence, x, y, w, h = detection
        #print(detection, name)
        if(name=='cell phone'):
            cx = x + (w / 2)
            cy = y + (h / 2)
            xmin_depth = x+round(w/4)
            ymin_depth = y+round(h/4)
            xmax_depth = x+round(w*.75)
            ymax_depth = y+round(h*.75)
            depth_phone = depth[ymin_depth:ymax_depth,xmin_depth:xmax_depth].astype(float)
            depth_phone_scaled = depth_phone * depth_scale
            dist,_,_,_ = cv2.mean(depth_phone_scaled)
            #print("Detected {0:.3} meters away.".format(dist))
            distanceString = "Detected {0:.3} meters away.".format(dist)
            #postCoords(dist)
                                 
        
            return True, detection, dist,inference_time_obj
            
    return False,False,0,inference_time_obj



pin = ""

counter = 0
currentSelect = ""

while True:
    
    start_time = time.time()
    
    color_image,depth = getIntelFrames()
     

    frame = color_image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    # Get data scale from the device and convert to meters
    
    
    width, height, inference_time, results = yolo.inference(frame)
    hp_startTime = time.time()
    frame,indexX,indexY = getFingers(frame)
    hp_endTime = time.time()
    
 #   phone_detected, cellObj, cellDist,inference_time_obj = cellPhoneDetection(frame, depth)#continuous cellphone detection
 #   if(phone_detected):
 #       color=(0,255,0)
 #       inference_time=inference_time_obj+inference_time

        
 #       distanceString = "Detected {0:.3} meters away.".format(cellDist)
 #       cv2.putText(frame, "phone "+distanceString, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
            
    
    if(cell_phone_flag):
        #if(distHand):
        #    postCoords(distHand,interactDist)
        
        if(True):
            id, name, confidence, x, y, w, h = interActionZone

            h= h+100 #scaling grid larger
            w=w+100

            for i in range(0, 4):

                rowy=i*h/3
                
                s1 = (int(x+w/3),int(y+rowy))
                e1 = (int(x+w*2/3),int(y+h/3+rowy))
                s2 = (int(x+w*2/3),int(y+rowy))
                e2 = (int(x+w),int(y+h/3+rowy))
                s3 = (int(x+w),int(y+rowy))
                e3 = (int(x+w*4/3),int(y+h/3+rowy))

                color = (255, 0, 0)
                colorSelect = (0, 255, 0)

                frame = cv2.rectangle(frame,s1 , e1, color, thickness)
                frame = cv2.rectangle(frame,s2 , e2, color, thickness)
                frame = cv2.rectangle(frame,s3 , e3, color, thickness)
                
                symbol = ""

                if (i == 3):
                    symbol = "<"
                else:
                    symbol = str(i * 3 + 1)
                if (indexX >= int(x+w/3) and indexX <= int(x+w*2/3) and indexY >= int(y+rowy) and indexY <= int(y+h/3+rowy)):
                    if (counter == 0 or currentSelect == symbol):
                        counter = counter + 1
                    else:
                        counter = 0
                    currentSelect = symbol
                    frame = cv2.putText(frame, symbol, (int(x+w*1.5/3), int(y+h/6+rowy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, colorSelect, 2, cv2.LINE_AA)
                else:
                    frame = cv2.putText(frame, symbol, (int(x+w*1.5/3), int(y+h/6+rowy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2, cv2.LINE_AA)
                if (i == 3):
                    symbol = "0"
                else:
                    symbol = str(i * 3 + 2)
                if (indexX >= int(x+w*2/3) and indexX <= int(x+w) and indexY >= int(y+rowy) and indexY <= int(y+h/3+rowy)):
                    if (counter == 0 or currentSelect == symbol):
                        counter = counter + 1
                    else:
                        counter = 0
                    currentSelect = symbol
                    frame = cv2.putText(frame, symbol, (int(x+w*2.5/3), int(y+h/6+rowy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, colorSelect, 2, cv2.LINE_AA)
                else:
                    frame = cv2.putText(frame, symbol, (int(x+w*2.5/3), int(y+h/6+rowy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2, cv2.LINE_AA)
                if (i == 3):
                    symbol = ">"
                else:
                    symbol = str(i * 3 + 3)
                if (indexX >= int(x+w) and indexX <= int(x+w*4/3) and indexY >= int(y+rowy) and indexY <= int(y+h/3+rowy)):
                    if (counter == 0 or currentSelect == symbol):
                        counter = counter + 1
                    else:
                        counter = 0
                    currentSelect = symbol
                    frame = cv2.putText(frame, symbol, (int(x+w*3.5/3), int(y+h/6+rowy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, colorSelect, 2, cv2.LINE_AA)
                else:
                    frame = cv2.putText(frame, symbol, (int(x+w*3.5/3), int(y+h/6+rowy)), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2, cv2.LINE_AA)

            if (counter == 15):
                if (currentSelect == "<"):
                    pin = pin[0:len(pin)-1]
                elif (currentSelect == ">"):
                    pin = ""
                else:
                    pin = pin + currentSelect

            frame = cv2.putText(frame, pin, (int(x+w/3), int(y+h*3/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, colorSelect, 2, cv2.LINE_AA)
                
            set_interaction =True
        

    else:
        if(set_interaction==False):
            #print("Interaction Zone Set")
            cell_phone_flag, interActionZone, cellDist,inference_time_obj = cellPhoneDetection(frame, depth)
            interactDist = cellDist-.6
            inference_time=inference_time_obj+inference_time


    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    if hands != -1:
        hand_count = int(hands)               
            
   

    # display hands
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        xmin_depth = x+round(w/4)
        ymin_depth = y+round(h/4)
        xmax_depth = x+round(w*.75)
        ymax_depth = y+round(h*.75)      
        
        
        # Crop depth data:
        depth_hand = depth[ymin_depth:ymax_depth,xmin_depth:xmax_depth].astype(float)
        depth_hand_scaled = depth_hand * depth_scale
        distHand,_,_,_ = cv2.mean(depth_hand_scaled)
        distanceString = "Detected {0:.3} meters away.".format(distHand)
        #showColorizedDepth(depth_frame,detection,distanceString)        
       
        # Display confidence and distance
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text+distanceString, (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    
    if indexX:
        indexDist=(depth[int(indexY),int(indexX)].astype(float))*depth_scale
        iDistanceSTR = "Detected {0:.3} meters away.".format(indexDist)
        cv2.putText(frame,"Index Finger "+ iDistanceSTR,(15,35),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2 )

    
    # display fps
    cv2.putText(frame, f'H & C {round(1/(inference_time),2)} FPS', (frame.shape[1]-200,55), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)
    cv2.putText(frame,f'HP {round(1/(hp_endTime - hp_startTime),2)} FPS',(frame.shape[1]-200,35),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2 )

    cv2.putText(frame,f'All {round(1/(time.time() - start_time),2)} FPS',(frame.shape[1]-200,15),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2 )
        
    cv2.imshow("preview", frame)  
    
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


pipeline.stop()
print('done')

# In[ ]:


pipeline.stop()


# In[ ]:


import pyrealsense2 as rs

#rs_serial='030522070959' #serial number for realsense camera
rs_serial ='031222070617'
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(rs_serial)
#config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)


print("[INFO] start streaming...")
profile = pipeline.start(config)
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
device.hardware_reset()
pipeline.wait_for_frames()
pipeline.stop()


# In[ ]:


#https://github.com/google/mediapipe/issues/2138
#For me the error got away after I uninstalled and reinstalled the opencv-contrib-python for 4.5.2
#MediaPipe doesn't require a specific opencv version: https://github.com/google/mediapipe/blob/master/requirements.txt#L4. Is it


# In[ ]:


obj_model.inference(frame)


# In[ ]:


frame.shape[1]


# In[ ]:


for detection in results_obj:
    id, name, confidence, x, y, w, h = detection
    #print(detection, name)
    if(name=='cell phone'):
        print(detection)


# In[ ]:


start_point+40


# In[ ]:


cellDist


# In[ ]:


0/3


# In[ ]:


cellObj


# In[ ]:


x


# In[ ]:


y


# In[ ]:


x


# In[ ]:


start


# In[ ]:


cellDist


# In[ ]:


pipeline.start()


# In[ ]:





# In[ ]:




