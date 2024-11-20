
shrimps - v2 2022-01-30 5:42pm
==============================

This dataset was exported via roboflow.com on August 8, 2022 at 10:52 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 289 images.
No are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random shear of between -25° to +25° horizontally and -27° to +27° vertically
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 10 percent of pixels


