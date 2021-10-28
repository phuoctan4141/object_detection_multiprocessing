#=================================================#
# Welcome to the object detection multiprocessing #
# Author: phuoctan4141                            #
# Date: 28/10/2021                                #
# Version: 0.3.33                                 #
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
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
#model link
model_path = 'refer_4a_version_1.tflite'

def grab_screen(p_input, f):

  cap = cv2.VideoCapture('video.mp4')

  # Check if camera opened successfully
  if (cap.isOpened() == False): 
    print("Error opening video stream or file")

  while(cap.isOpened()):

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
            for face_score,face_bbox_ratio in detector.run_inference(image_np):
                if len(face_bbox_ratio) > 0:
                    for bbox in face_bbox_ratio:
                        detector.draw_bbox(image_np, bbox, face_score)
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

      # Show image with detection
      cv2.imshow(title, image_np)
      # Bellow we calculate our FPS
      fps+=1
      TIME = time.time() - start_time
      print("FPS: ", fps / (TIME))
      fps = 0
      start_time = time.time()
      
      # Press "q" to quit
      if cv2.waitKey(25) & 0xFF == ord("q"):
        f.value = 1
        cv2.destroyAllWindows()
    else:
      image_np = p_output2.recv()
      continue
# end def

if __name__=="__main__":

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
