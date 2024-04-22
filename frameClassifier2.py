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


def file_classifier(input_path: str, output_path: str, emotions: dict, target_emo: str, id_map: dict):
    counter_dict = {}
    for level in os.listdir(input_path):
        # print(counter_dict)
        domain_path = os.path.join(input_path, level)

        for domain in tqdm(os.listdir(domain_path), desc=f'Processing {target_emo}'):
            # print(domain)
            domain_num = id_map[domain]
            # print(domain_num)
            if domain_num not in counter_dict.keys():
                counter = 1
                counter_dict[domain_num] = counter
            current_counter = counter_dict[domain_num]

            image_path = os.path.join(domain_path, domain)
            emotion_folders = os.listdir(image_path)
            for emotion_folder in emotion_folders:

                if emotion_folder in emotions.keys() and emotions[emotion_folder] == target_emo:
                    emotion_name = emotions[emotion_folder]
                    # print('input path')
                    # print(os.path.join(image_path, emotion_folder))
                    # print('output path')
                    # print(os.path.join(output_path, f'{domain_num:02}', emotion_name))
                    # print('current counter')
                    # print(current_counter)
                    current_counter = file_moving(os.path.join(image_path, emotion_folder),
                                                  os.path.join(output_path, f'{domain_num:02}', emotion_name),
                                                  current_counter)
            # print('counter after')
            # print(current_counter)
            counter_dict[domain_num] = current_counter


def main():
    # target_emotions = ['angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    target_emotions = {'angry': 'angry', 'disgusted': 'disgust', 'fear': 'fear', 'happy': 'happy', 'sad': 'sad', 'surprised': 'surprise'}
    source_path = 'dataTmp/MEAD-frontOnly-frames-processed2'
    target_path = 'dataTmp/expData-processed'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    domain_actor_map = {}
    id_counter = 1
    for level in os.listdir(source_path):
        for domain in os.listdir(os.path.join(source_path, level)):
            if domain not in domain_actor_map.keys():
                domain_actor_map[domain] = id_counter
                id_counter += 1

    # print(domain_actor_map)

    get_angry = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 'angry', domain_actor_map))
    get_disgust = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 'disgust', domain_actor_map))
    get_fear = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 'fear', domain_actor_map))
    get_happy = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 'happy', domain_actor_map))
    get_sad = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 'sad', domain_actor_map))
    get_surprise = Process(target=file_classifier, args=(source_path, target_path, target_emotions, 'surprise', domain_actor_map))

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
