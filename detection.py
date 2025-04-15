import face_recognition
import cv2
import pickle
import numpy as np
from threading import Thread
import queue
import time
import dlib  # Required for CUDA check

# =============================================
# KONFIGURASI UTAMA
# =============================================
RTSP_URL = "rtsp://admin:S3mangat45**@192.168.1.64?tcp"  # Gunakan TCP
SKIP_FRAMES = 2  # Process setiap 3 frame
SHOW_FPS = True
MIN_FACE_SIZE = 100  # Ukuran minimal wajah (pixel)

# Check CUDA availability
CUDA_AVAILABLE = dlib.DLIB_USE_CUDA
MODEL_DETECTION = "cnn" if CUDA_AVAILABLE else "hog"  # Gunakan CNN jika CUDA tersedia

print(f"Running with {'CUDA' if CUDA_AVAILABLE else 'CPU'} acceleration")
print(f"Using {MODEL_DETECTION.upper()} model for detection")

# =============================================
# VIDEO STREAM CLASS (Optimized)
# =============================================
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
    start_time = time.time()

    while True:
        frame = cap.read()
        frame_count += 1
        
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue
            
        # Convert to RGB and resize
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # STEP 1: FACE DETECTION (Optimized)
        face_locations = face_recognition.face_locations(
            rgb_frame,
        )
        
        if face_locations:
            # STEP 2: FACE ENCODING (Optimized)
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations,
                num_jitters=1
            )

            # STEP 3: FACE RECOGNITION
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
                
                # Visualization
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                # Additional face info
                cv2.putText(frame, f"Size: {right-left}x{bottom-top}", 
                          (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Display FPS and acceleration info
        if SHOW_FPS:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            accel_type = "CUDA" if CUDA_AVAILABLE else "CPU"
            cv2.putText(frame, f"FPS: {fps:.1f} | {accel_type} | {MODEL_DETECTION.upper()}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Face Detection + Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()