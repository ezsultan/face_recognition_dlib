import face_recognition
import cv2
import pickle
import numpy as np
from threading import Thread
import queue
import time
import dlib

RTSP_URL = "rtsp://admin:S3mangat45**@192.168.1.64?tcp"
SKIP_FRAMES = 2  # Process setiap 3 frame
SHOW_FPS = True
MIN_FACE_SIZE = 100  # Ukuran minimal wajah (pixel)

# Check CUDA availability
CUDA_AVAILABLE = dlib.DLIB_USE_CUDA
MODEL_DETECTION = "cnn" if CUDA_AVAILABLE else "hog"  # Gunakan CNN jika CUDA tersedia

print(f"Running with {'CUDA' if CUDA_AVAILABLE else 'CPU'} acceleration")
print(f"Using {MODEL_DETECTION.upper()} model for detection")

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 15)
        self.stopped = False
        self.Q = queue.Queue(maxsize=1)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.reconnect()
                continue
                
            frame = cv2.resize(frame, (640, 360))  # Downscale
            if self.Q.empty():
                self.Q.put(frame)

    def reconnect(self):
        print("Reconnecting...")
        self.stream.release()
        time.sleep(2)
        self.stream = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    def read(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

# =============================================
# LOAD REGISTERED FACES
# =============================================
print("Loading known faces...")
with open("encodings.pkl", 'rb') as f:
    known_faces = pickle.load(f)

known_encodings = list(known_faces.values())
known_names = list(known_faces.keys())

# =============================================
# MAIN PROCESSING
# =============================================
def main():
    cap = VideoStream(RTSP_URL).start()
    time.sleep(1.0)  # Warm-up
    
    frame_count = 0
    while True:
        frame = cap.read()
        frame_count += 1
        
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(
            rgb_frame,
        )
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations,
                num_jitters=1
            )

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    known_encodings,
                    face_encoding,
                    tolerance=0.5
                )
                
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_idx = np.argmin(face_distances)
                
                if matches[best_match_idx]:
                    name = known_names[best_match_idx]
                    confidence = 1 - face_distances[best_match_idx]
                    name = f"{name} ({confidence:.2f})"
                
                print(name, "detected person")
        
if __name__ == "__main__":
    main()