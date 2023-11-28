from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import os

output_dir = "output"
input_dir = "Dataset/Video_Song_Actor_01/Actor_01/01-02-01-01-01-01-01"
detector = MTCNN()

def calculate_affine_matrix(image, left_eye, right_eye):
    # calculate the center point between two eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    # calculate the angle between two eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # calculate the matrix for affine transformation
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    return M

def transform_box(M, box):
    x, y, width, height = box
    points = np.array([
        [x, y],
        [x + width, y],
        [x, y + height],
        [x + width, y + height]
    ])

    # 将点转换为仿射变换所需的格式
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    # 应用仿射变换
    transformed_points = M.dot(points_ones.T).T

    # 计算新的边框
    x_new = min(transformed_points[:, 0])
    y_new = min(transformed_points[:, 1])
    width_new = max(transformed_points[:, 0]) - x_new
    height_new = max(transformed_points[:, 1]) - y_new
    new_box = [int(x_new), int(y_new), int(width_new), int(height_new)]

    return new_box

def face_align(image, left_eye, right_eye):
    # calculate the matrix for affine transformation
    M = calculate_affine_matrix(image, left_eye, right_eye)

    # apply affine transformation
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

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
    
    # ajust the crop area to target size
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)

    return resized_image

def batch_processing(input_dir, output_dir):
    for frames in os.listdir(input_dir):
        if frames.endswith(".png") or frames.endswith(".jpg"):
            image_path = os.path.join(input_dir, frames)
            image = cv2.imread(image_path)
            
            result = detector.detect_faces(image)
            if result:
            # Should only have one face per image
                bounding_box = result[0]['box']
                keypoints = result[0]['keypoints']
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']
                
                # align face image
                aligned_image = face_align(image, left_eye, right_eye)
                aligned_box = transform_box(calculate_affine_matrix(image, left_eye, right_eye), bounding_box)
                processed_face = crop_and_resize_image(aligned_image, aligned_box)
                
                print(os.path.join(output_dir, f'processed_{frames}'))
                cv2.imwrite(os.path.join(output_dir, f'processed_{frames}'), processed_face)

def main():
    batch_processing(input_dir, output_dir)
    
if __name__ == "__main__":
    main()