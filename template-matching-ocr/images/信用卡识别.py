import cv2
import numpy as np


def show(img, name='demo'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取模板
temple = cv2.imread('ocr_a_reference.png')
show(temple)
ref = cv2.cvtColor(temple, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]  # threshold会返回两个参数，这里的[1]很细啊
show(ref)
# 轮廓检测
temple_Cnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(temple, temple_Cnts, -1, (0, 0, 255), 2)
show(temple)

digits_temp = []  # 方便排序，或许吧
digits = {}  # 初始化字典

# 这里使用的排序方法可绕过调库排序
# 首先，通过矩形轮廓boundingRect得到每一个矩形框的角落坐标，存进一个##一维数组##
for c in temple_Cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    digits_temp.append((x, y, w, h))
# 然后可以直接使用python内置sort函数将所有矩形按照位置排序
# 由排序后的矩形组再分别对图像切割缩放处理，比调库的方法简单许多
digits_temp.sort()
for (i, (gx, gy, gw, gh)) in enumerate(digits_temp):
    ROI = ref[gy:gy + gh, gx:gx + gw]
    ROI = cv2.resize(ROI, (57, 88))
    digits[i] = ROI


# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取待识别对象
img_1 = cv2.imread('credit_card_05.png')
img = cv2.imread('credit_card_05.png', 0)
img_1 = cv2.resize(img_1, (583, 368))
img = cv2.resize(img, (583, 368))
show(img)

# 礼帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectKernel)
show(tophat)
tophat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 索贝尔算子 梯度
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
gradX = np.abs(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 归一化
show(gradX)
gradY = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=3)
gradY = np.abs(gradY)
(minVal, maxVal) = (np.min(gradY), np.max(gradY))
gradY = (255 * ((gradY - minVal) / (maxVal - minVal)))  # 归一化
show(gradY)
sobelxy = cv2.addWeighted(gradY, 0.5, gradX, 0.5, 0)
show(sobelxy)

# 闭操作
sobelxy = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
show(sobelxy)
print(sobelxy)
sobelxy = cv2.threshold(sobelxy, 10, 255, cv2.THRESH_BINARY)[1]
show(sobelxy)
sobelxy = np.array(sobelxy, np.uint8)  # findContours函数只能处理uint8的数据类型
# 轮廓检测
soble_Cnts = cv2.findContours(sobelxy.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(img_1, soble_Cnts, -1, (0, 0, 255), 2)
show(img_1)

# 筛选轮廓
locs = []
locs_final = []
result = []
temp_max = 0
temp_num = 0
for c in soble_Cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if 2.5 < ar < 4.0:
        if 10 < h < 40 and 80 < w < 110:
            locs.append((x, y, w, h))
locs.sort()
# 截取卡号区域
output = []
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    group_1 = img_1[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
    group = img[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]
    print(group.shape)
    show(group)
    group_Cnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(group_1, group_Cnts, -1, (0, 0, 255), 2)
    show(group_1)
    for c in group_Cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        locs_final.append((x, y, w, h))
    locs_final.sort()
    for (fx, fy, fw, fh) in locs_final:
        ROI = group[fy: fy + fh, fx:fx + fw]
        ROI = cv2.resize(ROI, (57, 88))
        show(ROI)
        for j in range(10):
            match = cv2.matchTemplate(ROI, digits[j], cv2.TM_CCOEFF)
            score = cv2.minMaxLoc(match)[1]
            if score > temp_max:
                temp_max = score
                temp_num = j
        print(temp_num)
        result.append(temp_num)
        temp_max = temp_num = 0
    locs_final.clear()

print(result)
show(img_1)