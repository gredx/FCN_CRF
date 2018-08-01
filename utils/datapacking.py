# 把数据打包
from PIL import Image
from utils import  searchFile
import matplotlib.pyplot as plt
import numpy as np
import os,random
def get_randstr(len=20):
    bas_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    tar_str=''
    strlen = bas_str.__len__()
    for i in range(len):
        tar_str +=bas_str[random.randint(0,strlen-1)]
    return tar_str

def packImage(path,format='bmp',saved_name=None,size=(0,0,1000,800)):
    '''把指定目录下的图片打包成npz文件

    会把目录下文件和文件名一同打包
    r = np.load(file)   \n
    r[arr_0.npy] # 图片的包
    r[arr_1.npy] # 文件名的包


    :param path: 要打包图片的路径
    :param format: 图片的格式，默认是'bmp'
    :param saved_name: 输出文件的名字，缺省是一个长度为20的随机字符串
    :param size: a tuple.对图片裁剪的参数，(left,upper,right,lower),这个参数绝对不能超出图片的大小，否则出现意料之外的错误
    :return: 无返回值

    可能的异常结果:
        输出图片不正常：可能是参数size出错，查看是否超出了图片的大小
    '''
    lis = searchFile.search_spe_file(path,format)
    packfile =[]
    filesname =[]
    if saved_name is None:
        saved_name = get_randstr()
    for filepath in lis:

        filename = filepath.split('\\')[-1]
        filesname.append(filename)

        image = Image.open(filepath)
        image = image.crop(size)    # PIL 中的Image.crop参数是(left,upper,right,lower) 列，行
                                    # matplotplib.pyplot as plt 中的 plt.imread对象裁剪image[:800,:1000,:] 行，列
        image = np.array(image)
        packfile.append(image)

    np.savez(saved_name,packfile,filesname)

def unpackImage(file, imageSize, channel, output_dir=None):
    '''把npz图片还原为图片

    默认npz文件中arr_1是文件名，arr_0是图片，输入npz文件的路径，图片的尺寸，要输出的文件夹

    :param file: npz文件的路径
    :param imageSize: a tuple图片的尺寸 (hight,weight)
    :param channel: 图像的通道数，彩色3，灰度1
    :param output_dir: 输出文件夹的路径，若为空则创建一个随机名字的文件夹
    :return: 无返回值
    '''
    if output_dir is None:
        output_dir = get_randstr(20)
        os.mkdir(output_dir)
    os.chdir(output_dir)

    r = np.load(file)
    images = r['arr_0.npy']
    filesname = r['arr_1.npy']
    att1 = images.size // imageSize[0] // imageSize[1] // channel      # python3 中 / 结果是浮点数， //结果是整数
    images = np.reshape(images, (att1, imageSize[0], imageSize[1], channel))
    for image,name in zip(images,filesname):
        plt.imsave(name,image)


if __name__ == '__main__':
    path = 'D:\PythonCode\FCN_CRF\\utils\\trainData'
    saved_name = 'train.npz'
    packImage(path,size=(0,0,1200,800),saved_name=saved_name)
    unpackImage('D:\PythonCode\FCN_CRF\\train.npz', imageSize=(800,1200), channel=3)