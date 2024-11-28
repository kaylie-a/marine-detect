from src.marine_detect.predict import predict_on_images
import os

current_dir = os.getcwd()

predict_on_images(
    model_paths=[#current_dir+"\\models\\clam_model.pt", 
                 #current_dir+"\\models\\dolphin_model.pt", 
                 #current_dir+"\\models\\COTS_model.pt", 
                 #current_dir+"\\models\\fistularia_commersonii_model.pt"
                  current_dir+"\\models\\clownfish_model.pt",
                  current_dir+"\\models\\flounder_model.pt"],
    confs_threshold=[0.821, 0.747],
    images_input_folder_path=current_dir+"\\assets\\images\\input_folder",
    images_output_folder_path=current_dir+"\\assets\\images\\output_folder",
)
