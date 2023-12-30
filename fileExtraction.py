import subprocess
# import tarfile
import shutil
import os
import time


def extract_with_7zip(tar_path, target_subdir):
    # 使用 7-Zip 解压
    subprocess.call(['7z', 'x', tar_path, '-o' + target_subdir])


def extract_and_copy(input_dir: str, output_dir: str, tmp_dir: str, target_file: str):
    error_files = []
    for subdir in os.listdir(input_dir):
        print(subdir)
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path) and 'video.tar' in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, 'video.tar')

            start_time = time.time()
            # 解压 tar 文件
            # with tarfile.open(file_path, 'r') as tar:
            #     tar.extractall(path=tmp_dir)
            extract_with_7zip(file_path, tmp_dir)
            extraction_duration = time.time() - start_time
            print(f"Extracted '{file_path}' to '{tmp_dir}'. It took {extraction_duration:.2f} seconds.")
            # 复制指定文件夹
            source_folder = os.path.join(tmp_dir, target_file)
            destination_folder = os.path.join(output_dir, subdir)
            if os.path.exists(source_folder):
                shutil.copytree(source_folder, destination_folder)
            else:
                print(f"Folder '{target_file}' not found in '{subdir}'.")

            # 删除解压的文件夹（可选）
            shutil.rmtree(os.path.join(tmp_dir, 'video'))

        else:
            print('Not dir or video.tar')
            error_files.append(subdir)

    print(error_files)


def main():
    # 执行函数
    extract_and_copy(source_path, final_path, tmp_directory, folder_to_copy)


if __name__ == '__main__':
    # 路径和文件夹名称
    source_path = 'E:/MEAD'
    final_path = 'E:/MEAD-frontOnly'
    tmp_directory = 'tmp'  # 设置一个目标目录用于解压
    folder_to_copy = 'video/front'  # 替换为你想要复制的文件夹名称s

    # 创建目标目录（如果不存在）
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)
    if not os.path.exists(final_path):
        print('Your output directory does not exist.')
        quit()

    main()
