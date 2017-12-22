import os.path
import sys
import tensorflow as tf
import helper
import warnings
import scipy.misc
import numpy as np
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip
import helper_img_aug
import time

data_dir = './data'
runs_dir = './runs'
fcn_dir = './data/fcn-8'
num_classes = 2
#image_shape = (320, 1152)
image_shape = (160, 576)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)    
    
    return input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # one-by-ones

    l3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', strides=(1,1),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    l4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', strides=(1,1),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    l7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', strides=(1,1),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    #decoder
    
    #FCN-32 comes as is, with no SKIPS
    fcn_32 = tf.layers.conv2d_transpose(l7_1x1, num_classes, 4, strides=(2,2), padding='same',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    #FCN-16 SKIP from POOL-4 to the output
    fcn_16_input = tf.add(fcn_32, l4_1x1)
    fcn_16 = tf.layers.conv2d_transpose(fcn_16_input, num_classes, 4, strides=(2,2), padding='same',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    #FCN-8 adding one more SKIP from POOL-3
    fcn_8_input = tf.add(fcn_16, l3_1x1)
    fcn_8 = tf.layers.conv2d_transpose(fcn_8_input, num_classes, 16, strides=(8,8), padding='same',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        name='fcn8_out')
    
    return fcn_8
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes)) #from 4D to 2D
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for i in range(epochs):
        gen = get_batches_fn(batch_size)
        images, gt_images = next(gen)
        _, loss = sess.run([train_op, cross_entropy_loss], {
                        image_input:images, 
                        correct_label:gt_images,
                        keep_prob:0.6,
                        learning_rate:1e-3})
        print("Loss:{} at {} epoch.".format(loss, i))

tests.test_train_nn(train_nn)


def train(): #former run()
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    model_path = os.path.join(fcn_dir, str(time.time())) + ".ckpt"
    #builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        
        in_t, keep_prob, l3_t, l4_t, l7_t = load_vgg(sess, vgg_path)

        fcn_8 = layers(l3_t, l4_t, l7_t, num_classes)
        
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(fcn_8, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        train_nn(sess, 1000, 50, get_batches_fn, train_op, cross_entropy_loss, in_t,
             correct_label, keep_prob, learning_rate)
             
        #saving the model
        #builder.save()
        save_path = saver.save(sess, model_path)
        print("Model Saved in {}".format(model_path))

        #  Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, in_t)


def load_model(model_path, sess):
    model_path = os.path.join(fcn_dir, model_path)
    saver = tf.train.import_meta_graph(model_path + ".meta")
    saver.restore(sess, model_path)
    #tf.saved_model.loader.load(sess, ['FCN-8'], model_path)
    graph = tf.get_default_graph()
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    image_input = graph.get_tensor_by_name("image_input:0")
    fcn8_out = graph.get_tensor_by_name("fcn8_out/BiasAdd:0")
    logits = tf.reshape(fcn8_out, (-1, num_classes)) #from 4D to 2D
    return logits, keep_prob, image_input

def test_on_images(model_path):
    with tf.Session() as sess:
    #  Save inference data using helper.save_inference_samples
        logits, keep_prob, image_input = load_model(model_path, sess)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

def test_on_video(model_path, video_file_name):
    bitmask_buffer = []
    buffer_size = 20
    min_weight = 4
    with tf.Session() as sess:
        logits, keep_prob, image_input = load_model(model_path, sess)

        def process_frame(image):
            image = scipy.misc.imresize(image, image_shape)
            #img = image[370:690,64:1216,:]
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_input: [image]})#img]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            
            #movering average across few last frames
            bitmask_buffer.extend([np.dot(segmentation, 1)])
            if len(bitmask_buffer) > buffer_size:
                bitmask_buffer.pop(0)
            acc_mask = np.zeros(segmentation.shape, dtype=np.int64)
            for bm in bitmask_buffer:
                acc_mask += bm
            acc_mask[acc_mask[:,:,0]<min_weight] = 0           
            
            mask = np.dot(acc_mask, np.array([[0, 255, 0, 192]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, mask=mask, box=None)#(64,370))
            street_im = scipy.misc.imresize(street_im, (720, 1280))
            return np.array(street_im)
        
        clip_output = './video/project_video_output.mp4'
        clip = VideoFileClip(os.path.join("./video/", video_file_name))
        clip_processing = clip.fl_image(process_frame)
        clip_processing.write_videofile(clip_output, audio=False)


if __name__ == '__main__':
    tf.reset_default_graph()

    if len(sys.argv) == 1:
        print("Training...")
        train()
    elif sys.argv[1] == "aug":
        helper_img_aug.augment_images(os.path.join(data_dir, 'data_road/training'))
    elif len(sys.argv) == 3 and sys.argv[2] == "img":
        print("Testing on images...")
        test_on_images(sys.argv[1])
    elif len(sys.argv) == 3:
        print("Testing on video...")
        test_on_video(sys.argv[1], sys.argv[2])
    else:
        print("Missing arguments.")

