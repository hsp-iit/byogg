import cv2
import os
import argparse

def central_crop(frame, crop_width=640, crop_height=480):
    """
    Crops the given frame to the specified width and height from the center.
    """
    h, w = frame.shape[:2]
    # Calculate the starting point for the crop
    start_x = max((w - crop_width) // 2, 0)
    start_y = max((h - crop_height) // 2, 0)
    # Ensure that the crop region doesn't exceed the frame dimensions
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    return frame[start_y:end_y, start_x:end_x]

def process_video(input_video, output_folder, target_fps=30):
    """
    Extracts frames from the input video at a rate of `target_fps`,
    crops each frame to 640x480 using a central crop, and saves the frames to output_folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get the video's frames per second
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("Unable to determine video FPS. Exiting.")
        cap.release()
        return

    # Calculate how many frames to skip to achieve the target_fps
    frame_interval = int(round(video_fps / target_fps))
    if frame_interval < 1:
        frame_interval = 1

    print(f"Input video FPS: {video_fps:.2f}")
    print(f"Extracting one frame every {frame_interval} frames to achieve {target_fps} fps output.")

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Only process the frame if it's at the desired interval
        if frame_count % frame_interval == 0:
            # Crop the frame centrally to 640x480
            cropped_frame = central_crop(frame, 640, 480)
            # Save the frame with a padded filename for ordering
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, cropped_frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Processing complete. {saved_frame_count} frames saved in '{output_folder}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video, crop to 640x480, and save to a folder.")
    parser.add_argument("input_video", type=str, help="Path to the input video file")
    parser.add_argument("output_folder", type=str, help="Folder where extracted frames will be saved")
    args = parser.parse_args()

    process_video(args.input_video, args.output_folder)
