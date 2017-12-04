import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

class PhaseDetector:
    def __init__(self):
        self.model = Sequential()
        
    def init(self,input_shape,num_classes):
        self.model.add(Conv2D(
            32,
            kernel_size=(3,3),
            activation='relu',
            input_shape=input_shape
            ))
        self.model.add(Conv2D(
            64,
            kernel_size=(3,3),
            activation='relu'
        ))
        self.model.add(MaxPooling2D(
            pool_size=(2,2)
        ))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(num_classes,activation='softmax'))

        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy']
        )

    def fit(self,x_train,y_train,batch_size=16,epochs=100,verbose=0):
        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )

    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test,verbose=0)

    def predict(self,x):
        return self.model.predict(x)

    def loadmodel(self,filename):
        self.model.save(filename)

    def savemodel(self,filename):
        self.model = load_model(filename)
