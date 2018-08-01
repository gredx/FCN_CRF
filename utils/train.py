'''
训练U-net
'''
from keras.callbacks import *
from Unet import Unet
from utils.get_data import get_data
def train():
    # 读取训练数据和label数据
    imgs_train = get_data("train_data.npz",(512,512))
    imgs_label = get_data("train_label.npz",(512,512))
    # get Unet model
    model = Unet(512,512,loss=dice_coef_loss,metrics=dice_coef)
    # 保存权值
    checkPoint = ModelCheckpoint('weighs.h5',monitor='val_loss',save_best_only=True)
    # 开始训练
    print("start training.......")
    model.fit(imgs_train,imgs_label,batch_size=1,epochs=10,verbose=2,callbacks=[checkPoint],validation_split=0.1)


# 评估训练结果的函数
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

if __name__ == '__main__':
    train()