import os
import re
def search_file(tardir,filelist=[]):
    ' 搜索一个文件夹内所有文件,返回文件list'
    lis = os.listdir(tardir)
    for file in lis:
        filepath = os.path.join(tardir,file)
        if os.path.isdir(filepath):
            search_file(filepath,filelist)
        elif os.path.isfile(filepath):
             filelist.append(filepath)
    return filelist

def search_spe_file(tardir,format,filelist=[]):
    '搜索文件夹内指定格式的文件,返回文件list'
    filelist = search_file(tardir)
    list=[]
    regex = '(.*)'+format
    for file in filelist:
        if re.match(regex,file) :
            list.append(file)
    return list

if __name__ == '__main__':
    tar = os.getcwd()
    tar = 'D:\PythonCode\MyNet\图片打包'
    lis = search_spe_file(tar,'jpg')
    for file in lis:
        print(file.split('\\')[-1])