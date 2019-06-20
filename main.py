import os.path
import os
import json
import tornado.web
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
import tornado.options
import matplotlib.pyplot as plt  
import tensorflow as tf  
import numpy as np
import time
from PIL import Image
import cv2
from tensorflow.python.platform import gfile


TRAIN_FILE = './train_dir/model.pb'
INPUT_DATA = './flower_processed_data.npy'
# 下载的谷歌训练好的inception-v3模型文件名
MODEL_FILE = './inceptionV3/tensorflow_inception_graph.pb'
# inception-v3 模型中代表瓶颈层结果的张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
# 测试数据和验证数据比例。
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
strings = ['月季', '酢浆草', '万寿菊', '三色堇', '七姊妹']
def id_to_string(node_id):
    return strings[node_id]

def create_inception_graph():
    #加载已训练好的inception-v3模型
    with tf.Graph().as_default() as graph:
        with gfile.FastGFile(MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    return graph, bottleneck_tensor, jpeg_data_tensor

class MainHandler(tornado.web.RequestHandler):#加载网页
    def get(self):
        self.render("index.html")


class AjaxHandler(tornado.web.RequestHandler):#ajax数据接受和处理函数
    def post(self):
        data = self.get_argument("message")#获得图片字符串
        argb=[]
        temp=data.split(",")
        for i in range(len(temp)):#所有string转int，构成RGBA数组
            temp[i] = int(temp[i])
        for i in range(len(temp)):#对接收到的图片RGBA->RGB
            if (i+1)%4==0:
                pass
            else:
                argb.append(temp[i])#将数据存储到argb数组中

        imgs = np.array(argb, dtype=np.uint8)#list转变成array
        imgs=imgs.reshape((299,299,3))#处理图像格式

        a=Image.fromarray(imgs)  
        #a.show()#显示图像，查看是否得到准确图片
        a.save("./inceptionV3/predict_images/a.gif","GIF")#图片保存下来，接着再次读取，确保自己的图片格式准确
        
        graph, bottleneck_tensor, jpeg_data_tensor =create_inception_graph()#加载模型
        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE,bottleneck_tensor,jpeg_data_tensor)
            image_raw_data = gfile.FastGFile("./inceptionV3/predict_images/a.gif", 'rb').read()#读取图片
            image_value = sess.run(bottleneck_tensor,{jpeg_data_tensor:image_raw_data})
            image_value = np.squeeze(image_value)#转化成2048个一维特征向量
            with tf.gfile.FastGFile(TRAIN_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')#二进制模型加载
                with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    softmax_tensor = sess.graph.get_tensor_by_name('output/prob:0')
                    # 载入图片
                    predictions = sess.run(softmax_tensor, {'BottleneckInputPlaceholder:0': [image_value]})  # 图片格式是jpg格式
                    predictions = np.squeeze(predictions)  # 把结果转为1维数据
            
                    # 排序
                    top_k = predictions.argsort()[::-1]
                    print(top_k)
                    for node_id in top_k:
                        # 获取分类名称
                        human_string = id_to_string(node_id)
                        # 获取该分类的置信度
                        score = predictions[node_id]
                        print('%s (score = %.5f)' % (human_string, score))
                    self.write(id_to_string(top_k[0]))#返回数据结果
       

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/test", AjaxHandler),
    ],        
    static_path=os.path.join(os.path.dirname(__file__), "static"),

)

if __name__ == '__main__':
    application.listen(8000)
    tornado.ioloop.IOLoop.instance().start()
