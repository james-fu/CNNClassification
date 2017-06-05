#encoding=utf-8
import os
import tensorflow as tf

from PIL import Image
#
# cwd = os.getcwd()
cwd='/Users/Jessie/PycharmProjects/CNN/ep'
#
classes = {'n','p'}
# classes = {'normal','patient'}

#Make binary data
def create_record():
    writer = tf.python_io.TFRecordWriter("trainsize225data.tfrecords")
    for index, name in enumerate(classes):
        # class_path = cwd +"/"+ name+"/"
        class_path = cwd + '/'+name +'/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            # print img_path
            img = Image.open(img_path)
            img = img.resize((225, 225))
            img_raw = img.tobytes() # Convert the image to bytes
            print index,img_raw
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
               }))
            writer.write(example.SerializeToString())
    writer.close()

data = create_record()

#
# def read_and_decode(filename):
#     #根据文件名生成一个队列
#     filename_queue = tf.train.string_input_producer([filename])
#
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw' : tf.FixedLenFeature([], tf.string),
#                                        })
#
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [24, 24, 3])
#     img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#     label = tf.cast(features['label'], tf.int32)
#
#     return img, label
#
# img, label = read_and_decode2("catdata.tfrecords")
#
#
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                     batch_size=30, capacity=2000,
#                                                     min_after_dequeue=1000)
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(3):
#         val, l = sess.run([img_batch, label_batch])
#
#             # l = to_categorical(l, 12)
#         print(val.shape, l)
#
#         print val
#         print l




#show tfrecord image
filename_queue = tf.train.string_input_producer(["trainsize225data.tfrecords"]) #Read into the stream
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #Returns the file name and file
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #Get the feature object containing image and label
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [1225,225, 3])
label = tf.cast(features['label'], tf.int32)

with tf.Session() as sess: #Open a session
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(50):
        example, l = sess.run([image,label])#Get image and label in session
        img=Image.fromarray(example, 'RGB')
        img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#Save image
        print(example, l)
    coord.request_stop()
    coord.join(threads)