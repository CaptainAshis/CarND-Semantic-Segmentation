# Semantic Segmentation
### Introduction
This project capitalizes on pre-trained VGG-16 by converting it into a Fully Convolutional Network (FCN-8: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) and performaing Transfer Learning by making the FCN classify pixels of a road in front of a car.

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Usage
##### Image Augmentation

The training set contains only 289 training examples which is not that much to get a good grasp on challenging road conditions, in particular jumps in luminescence.
This is why, there is a way to generate additional training samples by augmenting (changing luminescence, translation, horizontal flipping, rotating, adding slat-and-pepper noise) the original images.

Run the following command to perform the image augmentation. Additional images and gt_images will be written in the same original folders:

```
python3 main.py aug
```


##### Training
Run the following command to train the project (999 epochs, 50 images per batch):
```
python3 main.py
```

In my tests, the resulting loss at the 1,000th epoch was 0.0297.
The results of training session will be written into a TF checkpoint with time-based name. You will see the name of the checkpoint upon training completion, e.g.:

```
...
Loss:0.031035281717777252 at 995 epoch.
Loss:0.03152341768145561 at 996 epoch.
Loss:0.024583401158452034 at 997 epoch.
Loss:0.027219999581575394 at 998 epoch.
Loss:0.02967345155775547 at 999 epoch.
Model Saved in ./data/fcn-8/1513930385.7582145.ckpt
```


##### Testing on Images

To run a trained model on images, you need to specify the checkpoint name as a parameter, followed by 'img' parameter, e.g.:

```
python3 main.py 1513930385.7582145.ckpt img
```

##### Testing on Video

To run a trained model on images, you need to specify the checkpoint name as a parameter, followerd by a mp4 filename located in  ./video folder, e.g.:

```
python3 main.py 1513930385.7582145.ckpt harder_challenge_video.mp4
```

the result will be written into ./video/project_video_output.mp4 file.
