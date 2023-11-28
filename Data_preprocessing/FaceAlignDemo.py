from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

image = cv2.imread("Dataset/Video_Song_Actor_01/Actor_01/01-02-01-01-01-01-01/frame_0002.png")

detector = MTCNN()

result = detector.detect_faces(image)
keypoints = result[0]['keypoints']
left_eye = keypoints['left_eye']
right_eye = keypoints['right_eye']
nose = keypoints['nose']
mouth_left = keypoints['mouth_left']
mouth_right = keypoints['mouth_right']

cv2.circle(image, left_eye, radius=5, color=(0, 255, 0), thickness=-1)
cv2.circle(image, right_eye, radius=5, color=(0, 255, 0), thickness=-1)
cv2.circle(image, nose, radius=5, color=(0, 255, 0), thickness=-1)
cv2.circle(image, mouth_left, radius=5, color=(0, 255, 0), thickness=-1)
cv2.circle(image, mouth_right, radius=5, color=(0, 255, 0), thickness=-1)
print(keypoints)
# 对于检测到的每个人脸，绘制边框
for face in result:
    # 获取边框坐标
    x, y, width, height = face['box']
    
    # 绘制矩形
    # 参数分别是：图像、左上角点、右下角点、颜色、线宽
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

def face_align(image, left_eye, right_eye):
    # calculate the center point between two eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    # calculate the angle between two eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # calculate the matrix for affine transformation
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

    # apply affine transformation
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return aligned_image

aligned_image = face_align(image, left_eye, right_eye)

cv2.imshow("Image with Eye Points", image)
cv2.imshow("Aligned Image", aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()