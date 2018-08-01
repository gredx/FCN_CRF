'''
训练U-net
'''
from keras.callbacks import *
from Unet import Unet
from utils.get_data import get_data
from keras.optimizers import Adam
def train():
    # 读取训练数据和label数据
    print("loading data.......")
    imgs_train = get_data("F:\liuyang\MyNet\\U-net\\train_data.npy",(512,512))
    imgs_label = get_data("F:\liuyang\MyNet\\U-net\\train_label.npy",(512,512))
    # get Unet model
    print("getting model...")
    model = Unet(512,512,optimizer=Adam(1e-4),loss=dice_coef_loss,metrics=dice_coef)
    # 保存权值
    checkPoint = ModelCheckpoint('weighs.h5',monitor='val_loss',save_best_only=True)
    # 开始训练
    print("start training.......")
    model.fit(imgs_train,imgs_label,batch_size=1,epochs=10,verbose=1, shuffle=True,callbacks=[checkPoint],validation_split=0.1)



if __name__ == '__main__':
    train()