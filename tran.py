import cv2
import os

# file_path = "image/test/Bach/6.jpg"
# image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('image/b61.jpg', image)

for path in os.listdir('image'):
    path_1 = os.path.join('image', path)
    path_tmp = os.path.join('image', (path + '1'))
    os.mkdir(path_tmp)
    for path_2 in os.listdir(path_1):
        path_3 = os.path.join(path_1, path_2)
        path_tmp2 = os.path.join(path_tmp, path_2)
        os.mkdir(path_tmp2)
        for path_4 in os.listdir(path_3):
            path_5 = os.path.join(path_3, path_4)
            path_tmp3 = os.path.join(path_tmp2, path_4)
            img = cv2.imread(path_5, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(path_tmp3, img)