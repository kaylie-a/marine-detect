from src.marine_detect.predict import predict_on_images
import os

current_dir = os.getcwd()

predict_on_images(
    model_paths=[current_dir+"\\models\\FishInv.pt", current_dir+"\\models\\MegaFauna.pt"],
    confs_threshold=[0.522, 0.6],
    images_input_folder_path=current_dir+"\\assets\\images\\input_folder",
    images_output_folder_path=current_dir+"\\assets\\images\\output_folder",
)
