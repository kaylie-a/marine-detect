
blue_ringed_octopus - v7 2023-12-05 1:37pm
==============================

This dataset was exported via roboflow.com on November 19, 2024 at 5:03 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 758 images.
Blue-ringed-octopus are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -45 and +45 degrees
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -5 and +5 degrees


