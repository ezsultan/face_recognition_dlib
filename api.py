from flask import Flask, request, jsonify
import face_recognition
import os
import pickle

app = Flask(__name__)

# Folder untuk menyimpan wajah yang terdaftar
registered_faces_folder = "registered_faces"
if not os.path.exists(registered_faces_folder):
    os.makedirs(registered_faces_folder)

# File untuk menyimpan encoding wajah
encodings_file = "encodings.pkl"

# Load encoding wajah yang sudah terdaftar (jika ada)
if os.path.exists(encodings_file):
    with open(encodings_file, 'rb') as f:
        registered_faces = pickle.load(f)
else:
    registered_faces = {}

# Endpoint untuk mendaftarkan wajah
@app.route('/register', methods=['POST'])
def register_face():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Simpan file yang diunggah
    file_path = os.path.join(registered_faces_folder, file.filename)
    file.save(file_path)
    
    # Load gambar dan dapatkan encoding wajah
    image = face_recognition.load_image_file(file_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) == 0:
        os.remove(file_path)
        return jsonify({"error": "No face detected in the image"}), 400
    
    # Simpan encoding wajah
    face_id = os.path.splitext(file.filename)[0]  # Nama file tanpa ekstensi
    registered_faces[face_id] = face_encodings[0]
    
    # Simpan encoding ke file
    with open(encodings_file, 'wb') as f:
        pickle.dump(registered_faces, f)
    
    return jsonify({"message": "Face registered successfully", "face_id": face_id}), 200

if __name__ == '__main__':
    app.run(debug=True)