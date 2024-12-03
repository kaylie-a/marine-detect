from src.marine_detect.predict import predict_on_images, predict_on_video
import os

current_dir = os.getcwd()

# Predict on all images in a folder
predict_on_images(
    model_paths = [ current_dir+"\\models\\!FishInv.pt",
                    current_dir+"\\models\\!MegaFauna.pt",
                    current_dir+"\\models\\clam_model.pt",
                    current_dir+"\\models\\clownfish_model.pt",
                    current_dir+"\\models\\dolphin_model.pt",
                    current_dir+"\\models\\fistularia_commersonii_model.pt",
                    current_dir+"\\models\\flounder_model.pt",
                    current_dir+"\\models\\leopard_coral_grouper_model.pt",
                    current_dir+"\\models\\lobster_model.pt",
                    current_dir+"\\models\\octopus_model.pt",
                    current_dir+"\\models\\rays3_model.pt",
                    current_dir+"\\models\\seahorse_model.pt",
                    current_dir+"\\models\\shark_model.pt",
                    current_dir+"\\models\\squid_model.pt" ],

    confs_threshold = [ 0.73,             # Fish & Invertebrates     - Given
                        0.73,             # MegaFauna & Rare Species - Given
          
                        0.85,             # Clam                      
                        0.69,             # Clownfish
                        0.73,             # Dolphin
                        0.87,             # Eagle Ray
                        0.49,             # Fistularia Commersonii
                        0.78,             # Flounder
                        0.75,             # Leopard Coral Grouper
                        0.70,             # Lobster
                        0.70,             # Octopus
                        0.70,             # 3 Rays
                        0.58,             # Seahorse
                        0.60,             # Shark
                        0.84 ],           # Squid

    images_input_folder_path  = current_dir+"\\assets\\images\\input_folder",
    images_output_folder_path = current_dir+"\\assets\\images\\output_folder",
)


# Predict on individual video
predict_on_video(
    # Less models for shorter compile time
    model_paths = [ current_dir+"\\models\\clownfish_model.pt",
                    current_dir+"\\models\\leopard_coral_grouper_model.pt",
                    current_dir+"\\models\\seahorse_model.pt" ],

    confs_threshold = [ 0.69,             # Clownfish
                        0.88,             # Leopard Coral Grouper
                        0.88 ],           # Seahorse

    input_video_path  = current_dir+"\\assets\\videos\\input_folder\\5548246-uhd_3840_2160_25fps.mp4",
    output_video_path = current_dir+"\\assets\\videos\\output_folder\\5548246-uhd_3840_2160_25fps.mp4",
)

