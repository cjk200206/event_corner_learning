import cv2
import numpy as np

def draw_matches(img1, keypoints1, img2, keypoints2, points1,points2):
    """
    在拼接后的图像上绘制描述子匹配结果

    参数：
    - img1: 第一幅图像
    - keypoints1: 第一幅图像的特征点坐标，形状为 (N, 2)
    - img2: 第二幅图像
    - keypoints2: 第二幅图像的特征点坐标，形状为 (N, 2)
    - matches: 匹配结果，形状为 (M, 2)，每个元素是两个特征点的索引

    返回：
    - matched_img: 绘制了匹配结果的拼接图像
    """

    # 拼接图像
    img1 = np.stack((img1,)*3, axis=-1)
    img2 = np.stack((img2,)*3, axis=-1)
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    matched_img = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    matched_img[:height1, :width1] = img1
    matched_img[:height2, width1:width1 + width2] = img2

    # 绘制检测结果
    for point1,point2 in zip(points1,points2):
        
        pt1 = (int(point1[1]), int(point1[0]))
        pt2 = (int(point2[1] + width1), int(point2[0]))  # 调整第二幅图像的坐标
            # 绘制特征点
        cv2.circle(matched_img, pt1, 5, (0, 255, 0), -1)
        cv2.circle(matched_img, pt2, 5, (0, 255, 0), -1)


    # 绘制匹配结果
    for keypoint1,keypoint2 in zip(keypoints1,keypoints2):
        
        pt1 = (int(keypoint1[1]), int(keypoint1[0]))
        pt2 = (int(keypoint2[1] + width1), int(keypoint2[0]))  # 调整第二幅图像的坐标
        
        # 绘制连线
        cv2.line(matched_img, pt1, pt2, (255, 0, 0), 2)

    return matched_img

# # 示例用法
# img1 = cv2.imread('image1.jpg')  # 读取第一幅图像
# img2 = cv2.imread('image2.jpg')  # 读取第二幅图像

# # 假设有 10 个特征点
# keypoints1 = np.random.rand(10, 2) * [img1.shape[1], img1.shape[0]]  # 第一幅图像的特征点
# keypoints2 = keypoints1 + np.random.randn(10, 2) * 20  # 第二幅图像的特征点，加一些噪声

# # 假设所有特征点都匹配
# matches = [(i, i) for i in range(10)]  # 匹配结果

# matched_img = draw_matches(img1, keypoints1, img2, keypoints2, matches)

# # 显示匹配结果图像
# cv2.imshow('Feature Matches', matched_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 或者保存匹配结果图像
# cv2.imwrite('matched_result.jpg', matched_img)
