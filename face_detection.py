import face_recognition
import cv2
import pickle

# Load encoding wajah yang sudah terdaftar
with open("encodings.pkl", 'rb') as f:
    registered_faces = pickle.load(f)

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Ubah frame ke RGB (face_recognition menggunakan RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi wajah dalam frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Bandingkan dengan wajah yang terdaftar
        match_found = False
        match_id = None
        for face_id, registered_encoding in registered_faces.items():
            match = face_recognition.compare_faces([registered_encoding], face_encoding)
            if match[0]:
                match_found = True
                match_id = face_id
                break
        
        # Gambar bounding box dan nama
        color = (0, 255, 0) if match_found else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, match_id if match_found else "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Tampilkan frame
    cv2.imshow('Realtime Face Matching', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()