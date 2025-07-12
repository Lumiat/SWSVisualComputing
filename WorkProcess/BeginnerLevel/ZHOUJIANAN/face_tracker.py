
import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def draw_chinese_text(img, text, pos, color=(255,255,255), size=20):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("simhei.ttf", size, encoding="utf-8")
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class KalmanFilter:
    def __init__(self, n_points):
        self.kalman_filters = []
        for _ in range(n_points):
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1
            self.kalman_filters.append(kf)

    def update(self, points):
        smoothed_points = []
        for i, point in enumerate(points):
            if i < len(self.kalman_filters):
                prediction = self.kalman_filters[i].predict()
                measurement = np.array([[point[0]], [point[1]]], np.float32)
                self.kalman_filters[i].correct(measurement)
                x = int(prediction[0].item())
                y = int(prediction[1].item())
                smoothed_points.append((x, y))
            else:
                smoothed_points.append(point)
        return smoothed_points

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=10,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        self.current_mode = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.kalman_filters = {}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process_facemesh(self, processed_frame, display_frame):
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = display_frame.shape

            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # 收集原始坐标并转为像素坐标
                raw_points = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

                # 初始化 Kalman 滤波器
                if i not in self.kalman_filters:
                    self.kalman_filters[i] = KalmanFilter(len(raw_points))

                # 平滑处理
                smoothed_points = self.kalman_filters[i].update(raw_points)

                # 绘制关键点
                for x, y in smoothed_points:
                    cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)  # 半径2比较合适

                # 手动绘制连接线（使用平滑后的坐标）
                for connection in self.mp_face_mesh.FACEMESH_TESSELATION:
                    start_idx, end_idx = connection
                    if start_idx < len(smoothed_points) and end_idx < len(smoothed_points):
                        pt1 = smoothed_points[start_idx]
                        pt2 = smoothed_points[end_idx]
                        cv2.line(display_frame, pt1, pt2, (255, 255, 0), 1)  # 线宽为1，颜色浅蓝

    def process_facedetection(self, processed_frame, display_frame):
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if results.detections:
            h, w, _ = display_frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                self.mp_drawing.draw_detection(
                    display_frame,
                    detection
                )

    def update_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def draw_info(self, frame):
        # 背景遮罩（先叠加）
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 再叠加文字
        cv2.putText(frame, f"Mode: {'FaceMesh' if self.current_mode == 0 else 'FaceDetection'}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 中文提示
        frame[:, :] = draw_chinese_text(
            frame, "按 M 切换模式，Q 退出", (10, frame.shape[0] - 50), (255, 255, 255), size=36)

def draw_mediapipe_landmarks(frame, face_landmarks_list, mp_face_mesh, mp_drawing, mp_drawing_styles):
    """
    使用 MediaPipe 官方样式绘制面部关键点和网格线
    :param frame: 要绘制的图像
    :param face_landmarks_list: MediaPipe 返回的 multi_face_landmarks
    :param mp_face_mesh: mp.solutions.face_mesh
    :param mp_drawing: mp.solutions.drawing_utils
    :param mp_drawing_styles: mp.solutions.drawing_styles
    """
    for face_landmarks in face_landmarks_list:
        # 面部网格（网格连接）
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # 面部轮廓线（粗轮廓）
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )


def main():
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        processed_frame = detector.apply_clahe(frame.copy())

        if detector.current_mode == 0:
            detector.process_facemesh(processed_frame, display_frame)
        else:
            detector.process_facedetection(processed_frame, display_frame)

        detector.update_fps()
        detector.draw_info(display_frame)
        cv2.imshow("Face Tracking", display_frame)

        key = cv2.waitKey(1) & 0xFF
        print("键值：", key)
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('m') or key == ord('M'):
            detector.current_mode = 1 - detector.current_mode
            print(f"切换到 {'FaceDetection' if detector.current_mode == 1 else 'FaceMesh'} 模式")

        if cv2.getWindowProperty("Face Tracking", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()
