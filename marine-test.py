from src.marine_detect.predict import predict_on_images
import os

current_dir = os.getcwd()

predict_on_images(
    model_paths=[current_dir+"\\runs\\detect\\train2\\weights\\best.pt"],
    confs_threshold=[0.63],
    images_input_folder_path=current_dir+"\\assets\\images\\input_folder",
    images_output_folder_path=current_dir+"\\assets\\images\\output_folder",
)
