# encoding:utf-8
import os.path
import re
import tensorflow as tf  
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

name=[]
filepath="D:\\flower"

# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath,name):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.isdir(child):
            eachFile(child,name)
        else:
            name.append(child)

if __name__ == "__main__":
    eachFile(filepath,name)
    for picture in name:
        images = cv2.imread(picture)
        sp = images.shape#obtain the image shape
        sz1 = sp[0]#height(rows) of image
        sz2 = sp[1]#width(colums) of image
        a=int(int(sz1/2)-int(sz1/3)) 
        b=int(int(sz1/2)+int(sz1/3)) 
        c=int(int(sz2/2)-int(sz2/3)) 
        d=int(int(sz2/2)+int(sz2/3)) 
        small_img = images[a:b,c:d]
        small_img=cv2.resize(small_img,(299,299))#缩放
        cv2.imwrite(picture,small_img)

        img = tf.read_file(picture) #读取图片
        img_data = tf.image.decode_jpeg(img, channels=3) #解码，原图
        left_right=tf.image.flip_left_right(img_data)#左右翻转
        up_down=tf.image.flip_up_down(img_data)#上下翻转
        tran=tf.image.transpose_image(img_data)#对角线翻转
        light_saturatino=tf.image.random_brightness(img_data, max_delta=0.5)#随机亮度
        light_saturatino=tf.image.random_saturation(light_saturatino, 0, 5)#随机饱和度
        ran=tf.image.random_hue(light_saturatino, 0.5)#随机色相
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            savepath=picture
            savepath=savepath.replace("D","C")
            savepath=savepath[:-4]+"1"+savepath[-4:]
            im=Image.fromarray(img_data.eval())
            im.save(savepath)

            savepath=savepath[:-4]+"2"+savepath[-4:]
            im=Image.fromarray(left_right.eval())
            im.save(savepath)

            savepath=savepath[:-4]+"3"+savepath[-4:]
            im=Image.fromarray(up_down.eval())
            im.save(savepath)

            savepath=savepath[:-4]+"4"+savepath[-4:]
            im=Image.fromarray(tran.eval())
            im.save(savepath)

            savepath=savepath[:-4]+"5"+savepath[-4:]
            im=Image.fromarray(ran.eval())
            im.save(savepath)





