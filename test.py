import os
import platform
import shutil
import tarfile

input_dir = "Dataset/Video_Song_Actor_01/Actor_01/01-02-01-01-01-01-01"

tag = input_dir.split("-")[-5]

print(tag)

print(platform.system())

with tarfile.open('video.tar', 'r') as tar:
    tar.extractall(path='tmp')

source_folder = os.path.join('tmp', 'video/front')
destination_folder = os.path.join('target', 'video/front')

shutil.copytree(source_folder, destination_folder)

shutil.rmtree('tmp/video')
