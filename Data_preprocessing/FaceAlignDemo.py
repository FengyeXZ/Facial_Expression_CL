from mtcnn.mtcnn import MTCNN
import cv2

image = cv2.imread("output/01-02-01-01-01-01-01/frame_0002.png")

detector = MTCNN()
result = detector.detect_faces(image)

if result:
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

cv2.imshow("Image with Eye Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()