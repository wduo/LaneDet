import tensorflow as tf
import os
import matplotlib.pyplot as plt

def file_name(file_dir):   #来自http://blog.csdn.net/lsq2902101015/article/details/51305825
    for root, dirs, files in os.walk(file_dir):  #模块os中的walk()函数遍历文件夹下所有的文件
        print(root) #当前目录路径  
        print(dirs) #当前路径下所有子目录  
        print(files) #当前路径下所有非目录子文件  

def file_name2(file_dir):   #特定类型的文件
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':   
                L.append(os.path.join(root, file))  
    return L 

path = file_name2('imgs')


#以下参考http://blog.csdn.net/buptgshengod/article/details/72956846 (十图详解TensorFlow数据读取机制)
#以及http://blog.csdn.net/uestc_c2_403/article/details/74435286

#path2 = tf.train.match_filenames_once(path)
file_queue = tf.train.string_input_producer(path, shuffle=True, num_epochs=2) #创建输入队列  
image_reader = tf.WholeFileReader()  
key, image = image_reader.read(file_queue)  
image = tf.image.decode_jpeg(image)  

with tf.Session() as sess:  
#    coord = tf.train.Coordinator() #协同启动的线程  
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord) #启动线程运行队列  
#    coord.request_stop() #停止所有的线程  
#    coord.join(threads)  

    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)

    #print (type(image))  
    #print (type(image.eval()))  
    #print(image.eval().shape)
    for _ in path+path:
        plt.figure
        plt.imshow(image.eval())
        plt.show()
