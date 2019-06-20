# flower-recognize
recognize your flower in picture by using Inception-V3 model

1)	事先拍好五种种类共计500张花朵图片(在flower_backup.zip中)，接着调用my_flower_cut.py依次读取文件目录下所有图片，对每一张图片，把图片缩放成299*299大小并保存下原图、左右翻转、上下翻转、对角线翻转、亮度饱和度色相随机改变的5张图像，并保存下来。运行data_process.py，该程序通过Inception-V3模型对图像进行处理，把卷积层的节点向量输出和对应花朵类型，分成训练、测试、验证三组数据写入到flower_processed_data.npy文件中去。接着运行train.py，该程序构建了一个新的单层全连接神经网络处理“花”的分类问题，程序通过读取flower_processed_data.npy的数据对新的网络进行训练和测试
2)	在客户端，首先调用声明相关菜单class="clip-content"、图片文件class="img-clip"、隐藏菜单选项class="clip-action nav-bar nav-bar-tab hidden"。然后在image-clip.js中的osMixin函数中，我们对输入设备系统进行判断然后加入网页拍照功能，对图像使用image-clip.js进行剪裁之后，进行监听。当用户按下发送键之后，新建一个画板canvas，将图片数据的长宽进行判定之后，启用一个最大正方形剪裁，将画板数据写入到smallcontent中，就会启动ajax的发送。
3)	在服务器中，我们会使用AjaxHandler类来接收ajax数据到data中，接着对data这个字符串进行分片、转化成argb数组，由argb数组转成图片并保存。这之后，启动create_inception_graph函数加载inception-v3模型，将读取到的图片数据提取出2048个特征值。加载二进制模型，并使用模型判断结果，对所得的predictions结果排序后输出返回给用户。

更多细节可见项目内ppt
