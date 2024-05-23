import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取两幅图像
img1 = cv2.imread('log/event_superpoint_sae/input_img/00000001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('log/event_superpoint_sae/input_img_transformed/00000001.jpg', cv2.IMREAD_GRAYSCALE)

# 创建 ORB 特征检测器
orb = cv2.ORB_create()

# 在两幅图像中检测特征点和描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 使用 BFMatcher 对象匹配描述子
matches = bf.match(des1, des2)

# 将匹配结果按照距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 绘制匹配结果
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 将 OpenCV 图像格式转换为 Matplotlib 图像格式
img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

# 显示图像
plt.imshow(img_matches_rgb)
plt.axis('off')
plt.show()
