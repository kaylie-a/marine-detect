
white-sharks - v4 2022-05-07 9:49am
==============================

This dataset was exported via roboflow.ai on May 7, 2022 at 7:50 AM GMT

It includes 1688 images.
Shark-species are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -20 and +20 degrees
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 5.5 pixels
* Salt and pepper noise was applied to 6 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Random exposure adjustment of between -38 and +38 percent


