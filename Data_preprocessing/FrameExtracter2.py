import subprocess
import shutil
import os
from multiprocessing import Process, Queue
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
# import time
# from tqdm import tqdm


def calculate_affine_matrix(left_eye, right_eye):
    # calculate the center point between two eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # calculate the angle between two eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # calculate the matrix for affine transformation
    m = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    return m


def transform_box(m, box):
    x, y, width, height = box
    points = np.array([
        [x, y],
        [x + width, y],
        [x, y + height],
        [x + width, y + height]
    ])

    # transform points to the format required by affine transformation
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    # apply affine transformation
    transformed_points = m.dot(points_ones.T).T

    # calculate the new box
    x_new = min(transformed_points[:, 0])
    y_new = min(transformed_points[:, 1])
    width_new = max(transformed_points[:, 0]) - x_new
    height_new = max(transformed_points[:, 1]) - y_new
    new_box = [int(x_new), int(y_new), int(width_new), int(height_new)]

    return new_box


def face_align(image, left_eye, right_eye):
    # calculate the matrix for affine transformation
    m = calculate_affine_matrix(left_eye, right_eye)

    # apply affine transformation
    aligned_image = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))

    return aligned_image


def crop_and_resize_image(image, box, target_size=(112, 112)):
    x, y, width, height = box
    center_x, center_y = x + width // 2, y + height // 2

    # not beyond the image boundary
    crop_size = min(width, height, center_x, image.shape[1] - center_x, center_y, image.shape[0] - center_y)

    # calculate the crop area
    crop_x1 = max(center_x - crop_size // 2, 0)
    crop_y1 = max(center_y - crop_size // 2, 0)
    crop_x2 = min(center_x + crop_size // 2, image.shape[1])
    crop_y2 = min(center_y + crop_size // 2, image.shape[0])

    # cut the crop area
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # adjust the crop area to target size
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)

    return resized_image


def extract_frames(video_path, output_folder, fps=2):
    """
    使用 ffmpeg 提取视频帧。

    :param video_path: 视频文件的路径
    :param output_folder: 存储提取帧的文件夹路径
    :param fps: 提取帧的频率（每秒帧数）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = [
        'ffmpeg', '-loglevel', 'panic', '-i', video_path, '-vf', f'fps={fps}',
        f'{output_folder}/img_%d.png'
    ]
    subprocess.run(command, shell=True)


def frames_processor(frames_queue, face_detector: MTCNN, frames_dir: str, output_dir: str):
    while True:
        if not frames_queue.empty():
            print("Processing frames...")
            frames_folder = frames_queue.get()
            if frames_folder is None:
                break
            levels = os.listdir(frames_dir)
            for level in levels:
                if not os.path.exists(os.path.join(frames_dir, level, frames_folder)):
                    continue
                emotions = os.listdir(os.path.join(frames_dir, level, frames_folder))
                for emotion in emotions:
                    frames = os.listdir(os.path.join(frames_dir, level, frames_folder, emotion))
                    for frame in frames:
                        img_path = os.path.join(frames_dir, level, frames_folder, emotion, frame)
                        image = cv2.imread(img_path)
                        result = face_detector.detect_faces(image)

                        if result:
                            # Should only have one face per image
                            bounding_box = result[0]['box']
                            keypoints = result[0]['keypoints']
                            left_eye = keypoints['left_eye']
                            right_eye = keypoints['right_eye']

                            # align face image
                            aligned_image = face_align(image, left_eye, right_eye)
                            aligned_box = transform_box(calculate_affine_matrix(left_eye, right_eye),
                                                        bounding_box)
                            processed_face = crop_and_resize_image(aligned_image, aligned_box)

                            output_path = os.path.join(output_dir, level, frames_folder, emotion)
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)
                            cv2.imwrite(os.path.join(output_path, frame), processed_face)
            print(f"Processed frames from '{frames_folder}'.")


def frames_extractor(buffer_queue, frame_proc_queue, buffer_dir: str, target_dir: str, emotions: list):
    while True:
        if not buffer_queue.empty():
            print("Extracting frames...")
            video_folder = buffer_queue.get()
            if video_folder is None:
                frame_proc_queue.put(None)
                break
            classes = os.listdir(os.path.join(buffer_dir, video_folder))
            for emotion in classes:
                if emotion not in emotions:
                    continue
                levels = os.listdir(os.path.join(buffer_dir, video_folder, emotion))
                for level in levels:
                    videos = os.listdir(os.path.join(buffer_dir, video_folder, emotion, level))
                    for video in videos:
                        video_file = os.path.join(buffer_dir, video_folder, emotion, level, video)
                        output_dir = os.path.join(target_dir, level, video_folder, emotion)
                        extract_frames(video_file, output_dir)
            frame_proc_queue.put(video_folder)
            shutil.rmtree(os.path.join(buffer_dir, video_folder))
            print(f"Extracted frames from '{video_folder}'.")


def videos_loader(works_queue, buffer_queue, file_dir: str, buffer_dir: str):
    while True:
        if not works_queue.empty() and buffer_queue.qsize() < 3:
            print("Loading videos...")
            next_work = works_queue.get()  # 从总任务队列中提取一个任务
            # 如果提取到的任务是 None，表示所有任务已经完成
            if next_work is None:
                buffer_queue.put(None)
                break
            source_file_path = os.path.join(file_dir, next_work)
            if os.path.exists(source_file_path):
                shutil.copytree(source_file_path, os.path.join(buffer_dir, next_work))
                buffer_queue.put(next_work)
            else:
                print(f"File '{source_file_path}' not found.")


if __name__ == '__main__':
    print(os.path.normpath(os.path.join(os.getcwd(), '..')))
    source_dir = 'E:/MEAD-frontOnly'
    # source_dir = '../expData'
    tmp_dir = '../tmp'
    frames_tmp_dir = 'E:/MEAD-frontOnly-frames'
    # frames_tmp_dir = '../target'
    # frames_processed_dir = '../target-processed'
    frames_processed_dir = 'E:/MEAD-frontOnly-frames-processed'

    emotion_classes = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'surprised']
    video_folders = os.listdir(source_dir)
    print(video_folders)

    detector = MTCNN()

    # extract_frames(os.path.join(source_dir, video_folders[0], 'angry', 'level_1', '001.mp4'), '../tmp')

    fileloader_queue = Queue()
    extraction_queue = Queue()
    processor_queue = Queue()

    for folder in video_folders:
        fileloader_queue.put(folder)
    fileloader_queue.put(None)  # 用于终止进程

    extractor = Process(target=frames_extractor, args=(extraction_queue, processor_queue, tmp_dir, frames_tmp_dir,
                                                       emotion_classes))
    loader = Process(target=videos_loader, args=(fileloader_queue, extraction_queue, source_dir, tmp_dir))
    processor = Process(target=frames_processor, args=(processor_queue, detector, frames_tmp_dir, frames_processed_dir))

    extractor.start()
    loader.start()
    processor.start()

    extractor.join()
    loader.join()
    processor.join()

    print("All videos have been processed.")
