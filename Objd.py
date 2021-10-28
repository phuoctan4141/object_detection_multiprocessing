#=================================================#
# Welcome to the object detection multiprocessing #
# Author: minhnhut                                #
# Date: 28/10/2021                                #
# Version: 0.1.0                                  #
#=================================================#

import cv2
import numpy as np
import tensorflow as tf

class Objd:
    #__slots__ = ['interpreter','input_details','output_details'] --- reduce a bit Memory
    def __init__(self,objd_path):
        self.interpreter = tf.lite.Interpreter(model_path=objd_path)

        self.interpreter.allocate_tensors()
        self.interpreter.invoke()  # warmup
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run_inference(self, image,threshold=0.3):

        img = cv2.resize(image,(224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  
        _accept = scores >= threshold
        yield scores[_accept],boxes[_accept]
    
    def draw_bbox(self,image,bbox,score):
    
        shape = image.shape[:2]
        start_x,start_y,end_x,end_y = int(bbox[1]*shape[1]),int(bbox[0]*shape[0]),int(bbox[3]*shape[1]),int(bbox[2]*shape[0])
        cv2.rectangle(image,(start_x,start_y),(end_x,end_y),(0,255,0),2)

# End code
