# YOLO adaptation

Implementation using Tensorflow and Python of [YOLO algorithm](https://pjreddie.com/darknet/yolo/) for object detection on single channel images.

## How to use it
The program can be launched from a command line with a few arguments (`python main.py --help` for a more detailed explanation).
- There are three possibles actions for now: 
  - train : Training of the network given a dataset, batch size, learning rate etc. 
  - test : Test a single image and save an image `prediction.png` in the working directory.
  - score : Performs some scoring of the model given a labelled test dataset (see help for more details).

- A dataset is a txt file with a path to one training image (png or jpeg) at each line. For each image `/path/img0.png` there must be a label file `/path/img0.txt`. 
- Each label file is formated as follow:
  - One line per bounding box.
  - Each line is  `t i j h w` where `t` is the type of the object (for now there is only class 0 possible since the detector performs no classification between detected objects), `i,j` are the coordinates (matrix convention) of the center of the bounding box and `h,w` its height and width. Each of these value is relative to the image size: between 0 and 1.  


## Modifications and interpretation of YOLO article [[1]](https://arxiv.org/abs/1506.02640) [[2]](https://arxiv.org/abs/1612.08242)

- The output of the network is a modified version of the two versions of YOLO : instead of giving the coordinates of the center of the bounding boxes and their dimensions relative to the whole image, our network gives the coordinate of the center of the bounding boxes *relative to the cell size* and the dimension *relative to the whole image*. That way, each of the five outputs of a bounding box can be scaled between 0 and 1.
- The loss function is simpler as the classification part is dropped.
- In the loss function, the bounding box dimension part of the output is treated differently from the bounding box position part with a different scaling factor and not a square root. At first, the network was designed to find constant size objects in the image. 
- The network archtecture is way smaller.
