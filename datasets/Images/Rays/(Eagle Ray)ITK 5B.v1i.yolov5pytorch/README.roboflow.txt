
ITK 5B - v1 2023-05-24 9:20pm
==============================

This dataset was exported via roboflow.com on May 24, 2023 at 2:42 PM GMT

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

The dataset includes 639 images.
Aetomylus-nichofii are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 10 percent of the image
* Random rotation of between -15 and +15 degrees
* Random brigthness adjustment of between -10 and +10 percent
* Random exposure adjustment of between -15 and +15 percent


