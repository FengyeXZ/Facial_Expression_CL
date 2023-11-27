import os
import subprocess
from tqdm import tqdm


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
        f'{output_folder}/frame_%04d.png'
    ]
    subprocess.run(command, shell=True)

def extract_all_frames(video_path, output_folder):
    """
    使用 ffmpeg 提取视频的所有帧。

    :param video_path: 视频文件的路径
    :param output_folder: 存储提取帧的文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = [
        'ffmpeg', '-i', video_path,
        f'{output_folder}/frame_%04d.png'
    ]
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_videos_in_folder(folder_path, output_root, actor_folder_relevant):
    """
    处理指定文件夹中的所有视频文件。

    :param folder_path: 包含视频文件的文件夹路径
    :param output_root: 存放输出图片的根目录
    """
    for video_file in tqdm(os.listdir(folder_path)):
        video_path = os.path.join(folder_path, video_file)
        output_folder = os.path.join(output_root, actor_folder_relevant, os.path.splitext(video_file)[0])
        extract_frames(video_path, output_folder)

# def batch_extract_all_frames(video_folder, output_root):
#     """
#     批量提取视频文件夹中所有视频的所有帧。

#     :param video_folder: 存放视频文件的文件夹
#     :param output_root: 存放输出图片的根目录
#     """
#     for video_file in os.listdir(video_folder):
#         video_path = os.path.join(video_folder, video_file)
#         output_folder = os.path.join(output_root, os.path.splitext(video_file)[0])
#         # extract_frames(video_path, output_folder)
#         extract_all_frames(video_path, output_folder)
        
def batch_process_dataset(dataset_root, output_root):
    """
    批量处理数据集中的所有视频文件。

    :param dataset_root: 数据集的根目录
    :param output_root: 存放输出图片的根目录
    """
    for folder_name in tqdm(os.listdir(dataset_root)):
        # print(folder_name)
        if folder_name.startswith("Video_Song_Actor_") or folder_name.startswith("Video_Speech_Actor_"):
            actor_folder_suffix = "Actor_" + folder_name.split('_')[-1]
            # print(actor_folder_suffix)
            actor_folder = os.path.join(dataset_root, folder_name, actor_folder_suffix)
            actor_folder_relevant = os.path.join(folder_name, actor_folder_suffix)
            print(actor_folder)
            if os.path.isdir(actor_folder):
                process_videos_in_folder(actor_folder, output_root, actor_folder_relevant)

# 示例使用
dataset_root = 'E:\Dataset_Raw'  # 视频文件夹的路径
output_root = 'E:\Dataset_Frames_test'  # 输出图片的根目录
# batch_extract_all_frames(video_folder, output_root)
batch_process_dataset(dataset_root, output_root)