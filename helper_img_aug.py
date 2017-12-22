# from https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

import os
import re
from glob import glob
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import scipy
from math import ceil, floor
from math import pi
import cv2


IMAGE_SIZE_W = 1242
IMAGE_SIZE_H = 375

def augment_images(data_folder):
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

    cnt = 0
    for image_file in image_paths:    
        gt_image_file = label_paths[os.path.basename(image_file)]
        image = scipy.misc.imresize(scipy.misc.imread(image_file), [IMAGE_SIZE_H, IMAGE_SIZE_W])
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), [IMAGE_SIZE_H, IMAGE_SIZE_W])
        data = [image, gt_image]
        imgs = np.array(data, dtype = np.float32)
        
        print("Scaling...")
        scales = [0.90, 0.75, 0.60]
        scaled_imgs = central_scale_images(imgs, scales)
        i = 0
        for img in scaled_imgs:
            fname = image_file + "_scaled_" + str(scales[i]) + ".png" \
                if i<len(scales) else gt_image_file + "_scaled_" + str(scales[i-len(scales)]) + ".png"
            #print(fname)
            scipy.misc.imsave(fname, img)
            i += 1
            
            
        print("Translating...")
        translated_imgs = translate_images(imgs)
        i = 0
        for img in translated_imgs:
            fname = (image_file if i%2==0 else gt_image_file) + "_translated_" + str(i//2) + ".png"
            #print(fname)
            scipy.misc.imsave(fname, img)
            i += 1
            
            
        print("Rotating...")
        rotated_imgs = rotate_images(imgs, -5, 5, 5)            
        i = 0
        for img in rotated_imgs:
            fname = (image_file if i%2==0 else gt_image_file) + "_rotated_" + str(i//2) + ".png"
            #print(fname)
            scipy.misc.imsave(fname, img)
            i += 1
            
        print("Flipping horizontally...")
        flipped_images = flip_images(imgs)            
        i = 0
        for img in flipped_images:
            fname = (image_file if i%2==0 else gt_image_file) + "_flipped_" + str(i//2) + ".png"
            #print(fname)
            scipy.misc.imsave(fname, img)
            i += 1
            
        print("Noise...")
        salt_pepper_noise_img = add_salt_pepper_noise(imgs)
        scipy.misc.imsave(image_file + "_noise.png", salt_pepper_noise_img[0])
        scipy.misc.imsave(gt_image_file + "_noise.png", imgs[1])

        print("Light...")
        change_light_imgs = change_light(imgs, 15)
        i = 0
        for img in change_light_imgs:
            if i%2==0: 
                scipy.misc.imsave(image_file + "_light_" + str(i//2) + ".png", img)
            else:
                scipy.misc.imsave(gt_image_file + "_light_" + str(i//2) + ".png", imgs[1])
            i += 1
            
        cnt +=1
        print("Processed: {}/{}".format(cnt, len(image_paths)))
        
    print("Done.")

#*******************************************************************

def change_light(X_imgs, num_out):
    imgs=[]
    for i in range(num_out):
        for img in X_imgs:
            #hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            #hsv[:,:,2] += (np.random.random() - .5) * 512.0
            #img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            YCrCb[:,:,0] += (np.random.random() - .5) * 256.0
            YCrCb[:,:,0] = YCrCb[:,:,0].clip(0.0, 255.0)
            img = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)
            imgs.append(img)
    imgs = np.array(imgs, dtype = np.float32)
    return imgs
  
#gaussian_noise_imgs = add_gaussian_noise(X_imgs)

#*******************************************************************

def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 255

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy
  
#salt_pepper_noise_imgs = add_salt_pepper_noise(X_imgs)


#*******************************************************************

def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE_H, IMAGE_SIZE_W, 3))
    tf_img1 = tf.image.flip_left_right(X)
    #tf_img2 = tf.image.flip_up_down(X)
    #tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            #flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            flipped_imgs = sess.run([tf_img1], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip
	
#flipped_images = flip_images(X_imgs)

#*******************************************************************


def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE_H, IMAGE_SIZE_W, 3))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    a = np.array(X_rotate, dtype = np.float32)
    a[((a[:,:,:,0]==0) & (a[:,:,:,1]==0) & (a[:,:,:,2]==0))] = [255,0,0]
    return a
	
# Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images
#rotated_imgs = rotate_images(X_imgs, -90, 90, 14)
#*******************************************************************

def get_translate_parameters(index):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE_H, ceil(0.8 * IMAGE_SIZE_W)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE_W))
        h_start = 0
        h_end = IMAGE_SIZE_H
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE_H, ceil(0.8 * IMAGE_SIZE_W)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE_W))
        w_end = IMAGE_SIZE_W
        h_start = 0
        h_end = IMAGE_SIZE_H
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE_H), IMAGE_SIZE_W], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE_W
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE_H)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE_H), IMAGE_SIZE_W], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE_W
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE_H))
        h_end = IMAGE_SIZE_H 
        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4
    X_translated_arr = []
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE_H, IMAGE_SIZE_W, 3), 
				  dtype = np.float32)
            X_translated[:,:,:,0]=255.0 # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
			 w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr
	
#translated_imgs = translate_images(X_imgs)

#*******************************************************************
def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE_H, IMAGE_SIZE_W], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE_H, IMAGE_SIZE_W, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data
	
# Produce each image at scaling of 90%, 75% and 60% of original image.
# scaled_imgs = central_scale_images(X_imgs, [0.90, 0.75, 0.60])



