# dataset_generation.py

import os
import imageio
from facenet_pytorch import MTCNN
import torch
import cv2
import math

class FrameExtractor:
    def __init__(self, fraction_frame_count, client_ids):
        self.fraction_frame_count = fraction_frame_count
        self.client_ids = client_ids
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    def extract_frames_with_faces(self, video_file, output_directory):
        reader = imageio.get_reader(video_file, 'ffmpeg')
        total_frames = reader.count_frames()
        self.desired_frame_count = int(self.fraction_frame_count * total_frames)
        frame_interval = max(1, math.ceil(total_frames / self.desired_frame_count))
        print(f"[ total frame : {total_frames}], [desired frame : {self.desired_frame_count}], [frame_interval : {frame_interval}]")

        extracted_frames = 0

        for frame_count, frame in enumerate(reader):
            if extracted_frames >= self.desired_frame_count:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = frame
                boxes, _ = self.mtcnn.detect(frame_rgb)

                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        # Ensure valid bounding box coordinates
                        if x1 < 0 or y1 < 0 or x2 > frame_rgb.shape[1] or y2 > frame_rgb.shape[0]:
                            print(f"Skipping invalid bounding box {box} in frame {frame_count}")
                            continue

                        face = frame_rgb[y1:y2, x1:x2]
                        if face.size == 0:
                            print(f"Detected face region is empty in frame {frame_count}")
                            continue

                        face_resized = cv2.resize(face, (256, 256))
                        name = os.path.splitext(os.path.basename(video_file))[0].lower()
                        output_path = os.path.join(output_directory, f"{name}_{extracted_frames + 1}.jpg")
                        cv2.imwrite(output_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
                        extracted_frames += 1

    def process_all_videos(self, dataset_folder, output_base_directory):
        output_directory_real = os.path.join(output_base_directory, "real")
        output_directory_attack = os.path.join(output_base_directory, "attack")
        os.makedirs(output_directory_real, exist_ok=True)
        os.makedirs(output_directory_attack, exist_ok=True)

        for subfolder in os.listdir(dataset_folder):
            subfolder_path = os.path.join(dataset_folder, subfolder)
            if os.path.isdir(subfolder_path):
                for video_file in os.listdir(subfolder_path):
                    client_id = video_file.split('_')[1]
                    if client_id in self.client_ids:
                        video_file_path = os.path.join(subfolder_path, video_file)
                        if os.path.isfile(video_file_path) and video_file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                            if subfolder.lower() == 'real':
                                self.extract_frames_with_faces(video_file_path, output_directory_real)
                            elif subfolder.lower() == 'attack':
                                self.extract_frames_with_faces(video_file_path, output_directory_attack)

if __name__ == "__main__":
    dataset_folder = r"E:\msu-msfd dataset(norway)\scene01"

    # Train dataset
    train_client_ids = {"02", "03", "05", "06", "07", "08", "09", "11", "12", "21", "22", "34", "53", "54", "55"}
    output_base_directory = r"E:\fasdataset_best\train"
    desired_frame_count = 0.3
    frame_extractor = FrameExtractor(desired_frame_count, train_client_ids)
    frame_extractor.process_all_videos(dataset_folder, output_base_directory)

    # Test dataset
    test_client_ids = {"01", "13", "14", "23", "24", "26", "28", "29", "30", "32", "33", "35", "36", "37", "39", "42", "48", "49", "50", "51"}
    output_base_directory = r"E:\fasdataset_best\test"
    desired_frame_count = 0.4
    frame_extractor = FrameExtractor(desired_frame_count, test_client_ids)
    frame_extractor.process_all_videos(dataset_folder, output_base_directory)

    # Validation dataset (using a separate fraction of frames if needed)
    val_client_ids = {"01", "13", "14", "23", "24", "26", "28", "29", "30", "32", "33", "35", "36", "37", "39", "42", "48", "49", "50", "51"}  # Adjust if needed
    output_base_directory = r"E:\fasdataset_best\val"
    desired_frame_count = 0.05
    frame_extractor = FrameExtractor(desired_frame_count, val_client_ids)
    frame_extractor.process_all_videos(dataset_folder, output_base_directory)
