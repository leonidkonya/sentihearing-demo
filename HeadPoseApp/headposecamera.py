import cv2
import numpy as np

from kivy.uix.camera       import Camera
from kivy.graphics.texture import Texture
from kivy.utils            import platform
from kivy.logger           import Logger

from utils import crop_image, softmax, preprocess, draw_axis


class HeadPoseCamera(Camera):
    def __init__(self, **kwargs):
        super(HeadPoseCamera, self).__init__(**kwargs)
        self.index = 1
        #p4a doesn't have a recipe for dlib, so we're using a cascade instead
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #self.detector = dlib.get_frontal_face_detector()
        if platform == 'android':
            self.texture = Texture.create(size=self.resolution,
                                          colorfmt='rgb')
            self.texture_size = list(self.texture.size)

            from jnius import autoclass

            File = autoclass('java.io.File')
            Interpreter = autoclass('org.tensorflow.lite.Interpreter')
            InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
            Tensor = autoclass('org.tensorflow.lite.DataType')
            DataType = autoclass('org.tensorflow.lite.DataType')
            TensorBuffer = autoclass('org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
            ByteBuffer = autoclass('java.nio.ByteBuffer')
            HashMap = autoclass('java.util.HashMap')

            model = File('WHENet.tflite')
            options = InterpreterOptions()
            self.interpreter = Interpreter(model, options)
            self.interpreter.allocateTensors()
            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.output_shape_yaw   = self.interpreter.getOutputTensor(0).shape()
            self.output_shape_pitch = self.interpreter.getOutputTensor(1).shape()
            self.output_shape_roll  = self.interpreter.getOutputTensor(2).shape()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()
            
        elif platform == 'win':
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter('WHENet.tflite',
                                                   num_threads=None)
            self.interpreter.allocate_tensors()
        else:
            print("not windows or android, aborting")
            exit()

    def _camera_loaded(self, *largs):
        if platform == 'android':
            self.texture = Texture.create(size=self.resolution,
                                          colorfmt='rgb')
            self.texture_size = list(self.texture.size)
        elif platform == 'win':
            self.texture = self._camera.texture
            self.texture_size = list(self.texture.size)
        else:
            #checked for other systems at init, will add other options later
            pass

    def on_tex(self, *l):
        if platform == 'android':
            buf = self._camera.grab_frame()
            if buf is None:
                return
            frame = self._camera.decode_frame(buf)
        elif platform == 'win':
            ret, frame = self._camera._device.read()
        else:
            #other options checked at init (no other possible options)
            pass

        if frame is None:
            print("something went wrong while grabbing frame")

        buf = self.process_frame_new(frame)
        self.texture.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte')
        super(HeadPoseCamera, self).on_tex(*l)


    def process_frame_new(self,frame):
        #if android: rgb
        #if windows: bgr
        if platform == 'android':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif platform == 'win':
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        #faces = self.detector(frame,0)
        #cascade requires image to be grayscale
        faces = self.face_cascade.detectMultiScale(frame_gray,
                                                   scaleFactor=1.2,
                                                   minNeighbors = 5)
        
        if len(faces)>0:
            #grabs the first face, assumes there aren't further faces to process
            face = faces[0]
            #bbox = (face.top(),face.left(),face.bottom(),face.right())
            bbox = np.asarray([face[1],face[0],face[1]+face[3],face[0]+face[2]])
            cropped_img, y_min, x_min, y_max, x_max = crop_image(frame,bbox)
            cropped_img = preprocess(cropped_img)

            #the main thing happens here:
            yaw, pitch, roll = self.predict(cropped_img)
            

            draw_axis(frame,yaw,pitch,roll,
                      tdx=(x_min+x_max)/2,
                      tdy=(y_min+y_max)/2,
                      size=abs(x_max-x_min)//2)

        #kivy texture needs image to be in rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame.tostring()
    

    def predict(self,img):
        
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = np.array(idx_tensor, dtype=np.float32)
        idx_tensor_yaw = [idx for idx in range(120)]
        idx_tensor_yaw = np.array(idx_tensor_yaw, dtype=np.float32)

        #normalize and convert to float32
        img = preprocess(img)

        if platform == 'android':
            try:
                #mainly using java classes imported though jnius.autoclass
                Logger.debug("beginning android prediction part")
                input_bytes = ByteBuffer.wrap(img.tobytes())
                Logger.debug("input bytes type: "+str(type(input_bytes))+"\n")
                output_bytes_yaw   = TensorBuffer\
                                     .createFixedSize(self.output_shape_yaw,
                                                      self.output_type)
                output_bytes_pitch = TensorBuffer\
                                     .createFixedSize(self.output_shape_pitch,
                                                      self.output_type)
                output_bytes_roll  = TensorBuffer\
                                     .createFixedSize(self.output_shape_roll,
                                                      self.output_type)
                Logger.debug("successfully created output bytes")

                #in jnius, a python list converts to a java array
                inputs = [input_bytes]
                        
                outputs = HashMap()
                outputs.put(0,output_bytes_yaw.getBuffer().rewind())
                outputs.put(1,output_bytes_pitch.getBuffer().rewind())
                outputs.put(2,output_bytes_roll.getBuffer().rewind())

                Logger.debug("successfully created output HashMap")

                self.interpreter.runForMultipleInputsOutputs(inputs,outputs)

                Logger.debug("successfully ran interpreter")
                
                yaw_array   = np.reshape(np.array(output_bytes_yaw).getFloatArray(),
                                         self.output_shape_yaw)
                pitch_array = np.reshape(np.array(output_bytes_pitch).getFloatArray(),
                                         self.output_shape_pitch)
                roll_array  = np.reshape(np.array(output_bytes_roll).getFloatArray(),
                                         self.output_shape_roll)

                Logger.debug("successfully converted output back to python")

            except Exception as e:
                Logger.debug("exception occurred:\n")
                Logger.debug(str(e))

        elif platform == 'win':
            self.interpreter.set_tensor(self.interpreter\
                                        .get_input_details()[0]['index'],img)
            self.interpreter.invoke()

            yaw_array   = self.interpreter\
                              .get_tensor(self.interpreter\
                                          .get_output_details()[0]['index'])
            pitch_array = self.interpreter\
                              .get_tensor(self.interpreter\
                                          .get_output_details()[1]['index'])
            roll_array  = self.interpreter\
                              .get_tensor(self.interpreter\
                                          .get_output_details()[2]['index'])

        yaw_predicted   = np.sum(softmax(yaw_array)*idx_tensor_yaw, axis=1)*3-180
        pitch_predicted = np.sum(softmax(pitch_array)*idx_tensor, axis=1)*3-180
        roll_predicted  = np.sum(softmax(roll_array)*idx_tensor, axis=1)*3-180

        return yaw_predicted, pitch_predicted, roll_predicted
