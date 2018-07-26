
from keras.models import Model
from keras.layers import Conv2D,Conv2DTranspose,MaxPool2D,concatenate,Input,Dropout


def Unet(img_rows,img_cols,optimizer='sgd', loss='mse', metrics='acc'):
    inputs = Input((img_rows,img_cols,1))
    conv1 = Conv2D(32,(3,3),padding="same",activation='relu')(input)
    conv1 = Conv2D(32,(3,3),padding='same',activation='relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64,(3,3),padding='same',activation='relu')(pool1)
    conv2 = Conv2D(64,(3,),padding='same',activation='relu')(conv2)
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128,(3,3),padding='same',activation='relu')(pool2)
    conv3 = Conv2D(128,(3,3),padding='same',activation='relu')(conv3)
    pool3 = MaxPool2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(256,(3,3),padding='same',activation='relu')(pool3)
    conv4 = Conv2D(256,(3,3),padding='same',activation='relu')(conv4)
    pool4 = MaxPool2D(pool_size=(2,2))(conv4)

    conv5 = Conv2D(512,(3,3),padding='same',activation='relu')(pool4)
    conv5 = Conv2D(512,(3,3),padding='same',activation='relu')(conv5)

    up6 = concatenate(input=[Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(conv5),conv4],axis=3)
    conv6 = Conv2D(256,(3,3),padding='same',activation='relu')(up6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(256,(3,3),padding='same',activation='relu')(conv6)

    up7 = concatenate([Conv2DTranspose(128,(3,3),strides=(2,2),padding='same')(conv6),conv3],axis=3)
    conv7 = Conv2D(128,(3,3),padding='same',activation='relu')(up7)
    conv7 = Conv2D(128,(3,3),padding='same',activation='relu')(conv7)

    up8 = concatenate([Conv2DTranspose(64,(3,3),strides=(2,2),padding='same'),conv2],axis=3)
    conv8 = Conv2D(64,(3,3),padding='same',activation='relu')(up8)
    conv8 = Conv2D(64,(3,3),padding='same',activation='relu')(conv8)

    up9 = concatenate([Conv2DTranspose(32,(3,3),strides=(2,2),padding='same'),conv1],axis=3)
    conv9 = Conv2D(32,(3,3),padding='same',activation='relu')(up9)
    conv9 = Conv2D(32,(3,3),padding='same',activation='relu')(conv9)

    conv10 = Conv2D(1,(1,1),padding='same',activation='sigmoid')(conv9)
    model = Model(input=[inputs],output=[conv10])
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return model