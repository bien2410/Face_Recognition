**Ứng dụng nhận diện khuôn mặt trong video hoặc qua camera, cho phép thêm và xóa khuôn mặt**

Uncomment lệnh train_base() nằm cuối cùng trong file Tranning.py để huấn luyện và lưu lại mô hình.

Sau đó comment lại lệnh này.

Sửa file Detect.py: video_capture = cv2.VideoCapture(0) thành video_capture = cv2.VideoCapture('video.mp4) để nhận diện khuôn mặt đã có trong video.

Sửa thành 0 để nhận diện khuôn mặt qua camera của máy

Chạy file GUI.py

Tắt camera hay video bằng enter.

