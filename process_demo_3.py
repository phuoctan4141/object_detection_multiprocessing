#=================================================#
# Welcome to the object detection multiprocessing #
# Author: phuoctan4141                            #
# Date: 28/10/2021                                #
# Version: 0.1.0                                  #
#=================================================#

# # Imports
import sys
import multiprocessing
from multiprocessing import Pipe
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
#cap link
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened() == True): 
    print("Opening camera")
cap.release()
print('Releasing camera')

#model link
model_path = 'refer_4a_version_1.tflite'

def grab_screen(p_input):

  cap = cv2.VideoCapture('video.mp4')

  # Check if camera opened successfully
  if (cap.isOpened() == False): 
    print("Error opening video stream or file")

  while(cap.isOpened()):
    #Grab screen image
    ret, img = cap.read()

    if ret == True:
      img = np.array(img)
      p_input.send(img)
    else:
      break
    
  # When everything done, release the video capture object
  print('end cap')
  cap.release()
  cv2.destroyAllWindows()
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
    
    print('end tf')
# end def

def Show_image(p_output2):
  global start_time, fps, cap

  while True:
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
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  print('end show')
  cap.release()
  cv2.destroyAllWindows()
# end def

if __name__=="__main__":

    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()

    # creating new processes
    p1 = multiprocessing.Process(target=grab_screen, args=(p_input,))
    p2 = multiprocessing.Process(target=TensorflowDetection, args=(p_output,p_input2,))
    p3 = multiprocessing.Process(target=Show_image, args=(p_output2,))

    # starting our processes
    p1.start()
    p2.start()
    p3.start()

# end code