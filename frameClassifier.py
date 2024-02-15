"""
用来给第一步处理结束的RAVDESS数据集转换成规范的数据集格式
"""
import os
from multiprocessing import Process
import shutil
from tqdm import tqdm


def file_moving(input_path: str, output_path: str, start_num: int):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    counter = start_num
    for images in os.listdir(input_path):
        frame_path = os.path.join(input_path, images)
        shutil.copy(frame_path, os.path.join(output_path, f'{counter:d}.png'))
        counter += 1
    return counter


def file_classifier(input_path: str, output_path: str, emotions: dict, target_emo_num: int):
    counter_dict = {}
    print(os.listdir(input_path))
    for domain in tqdm(os.listdir(input_path), desc=f'Processing {emotions[target_emo_num]}'):
        domain_num = int(domain[-2:])
        # print(domain_num)
        if domain_num not in counter_dict.keys():
            counter = 1
            counter_dict[domain_num] = counter
        current_counter = counter_dict[domain_num]

        image_path = os.path.join(input_path, domain, domain[-8:])
        emotion_folders = os.listdir(image_path)
        for emotion_folder in emotion_folders:
            emotion_num = int(emotion_folder[7])

            if emotion_num == target_emo_num:
                emotion_name = emotions[emotion_num]
                current_counter = file_moving(os.path.join(image_path, emotion_folder), os.path.join(output_path,
                                                                                                     f'{domain_num:02}',
                                                                                                     emotion_name),
                                              current_counter)
        counter_dict[domain_num] = current_counter


def main():
    target_emotions = {5: 'angry', 7: 'disgust', 6: 'fear', 3: 'happy', 4: 'sad', 8: 'surprise'}
    source_path = 'dataTmp/Dataset_Frames_processed'
    target_path = 'dataTmp/expData-processed'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    get_angry = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 5))
    get_disgust = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 7))
    get_fear = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 6))
    get_happy = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 3))
    get_sad = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 4))
    get_surprise = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 8))

    get_angry.start()
    get_disgust.start()
    get_fear.start()
    get_happy.start()
    get_sad.start()
    get_surprise.start()

    get_angry.join()
    get_disgust.join()
    get_fear.join()
    get_happy.join()
    get_sad.join()
    get_surprise.join()


if __name__ == '__main__':
    main()
