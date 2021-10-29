#=================================================#
# Welcome to the object detection multiprocessing #
# Author: phuoctan4141                            #
# Date: 28/10/2021                                #
# Version: 1.03.33rs                              #
#=================================================#

# # Imports
import sys
import multiprocessing
from multiprocessing import Process, Pipe, Value
import time
import cv2
import numpy as np
from Objd import Objd

# title of our window
title = "Object Detection Multiprocessing"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
#model link
model_path = 'refer_4a_version_1.tflite'
#video link
video_test = 'video.mp4'

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (7, 30)
# fontScale
fontScale = 1
# Red color in BGR
color = (0, 255, 0)
# Line thickness of 2 px
thickness = 2

def grab_screen(p_input, f):

  cap = cv2.VideoCapture(video_test)

  # Check if camera opened successfully
  if (cap.isOpened() == False): 
    print("Error opening video stream or file")

  while(cap.isOpened()):
    
    #flag exit code
    if f.value == 1:
      break
    #Grab screen image
    ret, img = cap.read()

    if ret == True:
      img = np.array(img)
      p_input.send(img)
    else:
      break

  # When everything done, release the video capture object
  cap.release()
  print('end cap')
  sys.exit(0)
# end def

def TensorflowDetection(p_output, p_input2):
    #__init__#
    global start_time
    detector = Objd(objd_path=model_path)

    #__run_inference__#
    while True:

          start_time = time.time()
          image_np = p_output.recv()
        
          if image_np.size:
            for face_score, face_bbox_ratio in detector.run_inference(image_np):
                if len(face_bbox_ratio) > 0:
                    for sscore, bbox in zip(face_score, face_bbox_ratio):
                        detector.draw_bbox(image_np, bbox, sscore)
          else:
            continue

          # Send detection image to pipe2
          p_input2.send(image_np)          
# end def

def Show_image(p_output2, f):
  global start_time, fps

  while True:
    if flag.value != 1:
      image_np = p_output2.recv()

      # Bellow we calculate our FPS
      fps+=1
      TIME = time.time() - start_time

      # putText fps
      image_np = cv2.putText(image_np, format(fps / (TIME), '.2f'), 
                             org, font, fontScale, color, thickness, cv2.LINE_AA)

      # Show image with detection
      cv2.imshow(title, image_np)
      
      fps = 0
      start_time = time.time()

      # Press "q" to quit
      if cv2.waitKey(25) & 0xFF == ord("q"):
        f.value = 1
        cv2.destroyAllWindows()
    else:
      #waiting flag exit code
      image_np = p_output2.recv()
      continue
# end def

if __name__=="__main__":

    #flag exit code
    flag = Value('i', 0)
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()
    # creating new processes
    p1 = Process(target=grab_screen, args=(p_input,flag,))
    p2 = Process(target=TensorflowDetection, args=(p_output,p_input2,))
    p3 = Process(target=Show_image, args=(p_output2,flag,))

    # starting our processes
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
# end code
