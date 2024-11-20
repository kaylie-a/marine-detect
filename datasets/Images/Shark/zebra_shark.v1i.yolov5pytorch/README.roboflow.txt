
zebra_shark - v1 2021-12-14 5:22pm
==============================

This dataset was exported via roboflow.ai on December 14, 2021 at 11:23 AM GMT

It includes 725 images.
Zebrashark are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random rotation of between -15 and +15 degrees
* Random Gaussian blur of between 0 and 1.5 pixels
* Salt and pepper noise was applied to 4 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* 50% probability of vertical flip


