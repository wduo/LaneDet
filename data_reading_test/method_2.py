# 导入tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt; 

# 新建一个Session
with tf.Session() as sess:
    # 我们要读三幅图片A.jpg, B.jpg, C.jpg
    filename = ['imgs/3.jpg', 'imgs/21.jpg', 'imgs/cat.0.jpg']
    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=5)
    # reader从文件名队列中读数据。对应的方法是reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之后，才会开始填充队列
    threads = tf.train.start_queue_runners(sess=sess)
    '''
    i = 0
    while True:
        i += 1
        # 获取图片数据并保存
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
    '''

    i = 0
    while True:
        i += 1
        img_data_jpg = tf.image.decode_jpeg(value)
        img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
        plt.figure(i) #图像显示  
        plt.imshow(img_data_jpg.eval())
        plt.show()
