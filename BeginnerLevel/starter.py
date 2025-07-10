import cv2

# 加载人脸检测器和关键点模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 你可以根据原图大小设定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # 缩放图像用于显示
    display_size = (640, 480)
    resized = cv2.resize(frame, display_size)

    scale_x = display_size[0] / frame.shape[1]
    scale_y = display_size[1] / frame.shape[0]

    # 绘制人脸框（缩放坐标后画）
    for (x, y, w, h) in faces:
        x_r, y_r, w_r, h_r = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
        cv2.rectangle(resized, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 255, 0), 2)

    # 检测关键点并绘制
    if len(faces) > 0:
        _, landmarks = facemark.fit(gray, faces)
        for landmark in landmarks:
            for (x, y) in landmark[0]:
                x_r = int(x * scale_x)
                y_r = int(y * scale_y)
                cv2.circle(resized, (x_r, y_r), 2, (0, 0, 255), -1)  # radius = 2 比较合适

    # 显示结果
    cv2.imshow("Webcam", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
