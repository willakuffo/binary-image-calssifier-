#import models
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import pyttsx3
import cv2
from tkinter.filedialog import askopenfilename
#import matplotlib.pyplot as plt

class CNN_image_classifier:
    def __init__(self):
        #speech
        self.speechEngine = pyttsx3.init()


        self.image_target_size = (32,32)
        self.filter_no = 32
        self.kernel_size = (3,3)
        self.act_fun = 'relu'#activation function
        self.inputImageShape = (32,32,3) #corresponds to image size
        self.CNN_classifier = Sequential()# seuential classifier object

        #adding a CNN
        #input layer
        self.CNN_classifier.add(Conv2D(self.filter_no,self.kernel_size,input_shape = self.inputImageShape,activation =self.act_fun))
        self.CNN_classifier.add(MaxPooling2D(pool_size = (2,2)))#image pool size  =(2,2) pixels  to minimize image loss
        self.CNN_classifier.add(Flatten()) #perform flattening on pooled image pixls ino a 1d matrix(vector)
        #hidden layer
        self.CNN_classifier.add(Dense(units = 512,activation = self.act_fun)) #add a fully connected layer with n units of hidden nodes
        #output layer
        self.CNN_classifier.add(Dense(units = 1,activation = 'sigmoid'))

        self.CNN_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

#getting training data
    def train(self,trainingdata,testData):
        self.speechEngine.say('Initializing Training')
        self.speechEngine.runAndWait()
        trainDataGen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
        testDataGen =  ImageDataGenerator(rescale = 1./255)
        trainingDataset = trainDataGen.flow_from_directory(trainingdata,target_size = self.image_target_size,batch_size = 32,class_mode = 'binary')
        testDataset = testDataGen.flow_from_directory(testData,target_size = self.image_target_size,batch_size = 32,class_mode = 'binary')

        #fit data to model
        #number of images = 8000
        training_size = 200
        testData_size = 2000
        self.CNN_classifier.fit_generator(trainingDataset,steps_per_epoch =training_size,validation_data = testDataset,validation_steps = testData_size,epochs=8)
        #print(self.history)
        self.speechEngine.say('Training completed and successful. Saving trained model...')
        self.speechEngine.runAndWait()
        print('Saving model and weights...')

        self.CNN_classifier.save('catDog_classifier.h5')
        #self.visualize()

    #make prediction
    def predict(self,image):
        loaded_CNN_classifier = load_model('catDog_classifier.h5')
        testImage = Image.open(image)
        img = cv2.imread(image)
        cv2.imshow('testImage',img)
        cv2.waitKey(1500)
        testImage = testImage.resize(self.image_target_size,Image.ANTIALIAS)
        testImage = np.array(testImage)#image.img_to_array(testImage)
        testImage = np.expand_dims(testImage,axis = 0)
        prediction = loaded_CNN_classifier.predict(testImage)
        #train_set.class_indices
        print(prediction)
        if prediction[0][0]==1:
            result = 'dog'
        else:

            result = 'cat'
        self.speechEngine.say('prediction is a '+result)
        self.speechEngine.runAndWait()
        print(result)

        return result

    def selectWithFileBrowser(self):
        image  = askopenfilename()
        self.predict(image)
        return image

    ''' def visualize(self):
       # self.history = self.CNN_classifier.fit()
        # Plot training & validation accuracy values

        plt.plot(self.history['acc'])
        plt.plot(self.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()'''




#CNN = CNN_image_classifier()
#CNN.train('training_set','test_set')
#CNN.predict('test_set/dogs/dog.4014.jpg')
#CNN.predict('test_set/cats/cat.4007.jpg')
#CNN.selectWithFileBrowser()
#CNN.visualize()
