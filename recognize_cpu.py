import os
import sys
import time
import warnings
import argparse
import cv2
import numpy as np

# Suppress library deprecation warnings (e.g. skimage.transform.estimate)
warnings.filterwarnings("ignore", category=FutureWarning)

import insightface
from insightface.app import FaceAnalysis


class PersonRecognizer:
    def __init__(self, known_faces_dir="known_faces", model_name="buffalo_sc", threshold=0.45):
        """
        Face Recognizer optimized for CPU with dynamic real-time enrollment.
        :param known_faces_dir: Directory containing person folders with reference photos.
        :param model_name: 'buffalo_sc' (lightweight, fast CPU) or 'buffalo_l' (accurate).
        :param threshold: Cosine similarity cutoff threshold (default: 0.45).
        """
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.database = {}  # { "Name": [embedding_1, embedding_2, ...] }

        print(f"[*] Initializing InsightFace ({model_name}) on CPU...")
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        print("[+] Model loaded successfully!")

    def enroll_image(self, name: str, img_path: str):
        """Extract and store normalized face embedding for a person from an image file."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"[-] Could not read: {img_path}")
            return False

        faces = self.app.get(img)
        if len(faces) == 0:
            print(f"[-] No face detected in: {img_path}")
            return False

        # If multiple faces are in the enrollment picture, choose the largest one
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.embedding / np.linalg.norm(face.embedding)

        if name not in self.database:
            self.database[name] = []
        self.database[name].append(embedding)
        print(f"    -> Enrolled 1 face photo for '{name}' ({os.path.basename(img_path)})")
        return True

    def load_known_faces(self):
        """Scan known_faces directory and enroll all persons."""
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)
            print(f"[!] Created directory '{self.known_faces_dir}'. Place person photo folders inside!")
            return

        print(f"[*] Loading reference photos from '{self.known_faces_dir}'...")
        total_enrolled = 0
        for person_name in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_path):
                print(f"[*] Scanning photos for: {person_name}")
                for filename in os.listdir(person_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(person_path, filename)
                        if self.enroll_image(person_name, img_path):
                            total_enrolled += 1

        print(f"[+] Finished loading. Total {total_enrolled} reference faces across {len(self.database)} people: {list(self.database.keys())}")

    def enroll_live_person(self, name: str, cap, window_name: str = "InsightFace - Realtime Recognition", rotation: int = 0):
        """
        Interactively captures face photos from webcam with manual SPACEBAR trigger
        and [y/n] acceptance verification to ensure high quality and diverse angles.
        Supports both new persons and expanding reference photos for existing persons.
        """
        name = name.strip()
        if not name:
            print("[!] Invalid name provided. Enrollment cancelled.")
            return False

        is_existing = name in self.database
        person_folder = os.path.join(self.known_faces_dir, name)
        os.makedirs(person_folder, exist_ok=True)

        print(f"\n" + "=" * 60)
        if is_existing:
            print(f"[*] Adding extra training photos for existing person: '{name}' (Currently {len(self.database[name])} photos)")
        else:
            print(f"[*] Enrolling new person: '{name}'")
        print("[*] INSTRUCTIONS:")
        print("    1. Look at the camera at desired angle (Front, Profile, Tilt, Smile).")
        print("    2. Press [SPACE] or [c] to CAPTURE a photo.")
        print("    3. Press [y] to ACCEPT and save the photo, or [n] to RETAKE.")
        print("    4. Press [r] to ROTATE orientation (0° -> 90° -> 180° -> 270°).")
        print("    5. Press [q] or [ESC] when you have captured enough angles.")
        print("=" * 60 + "\n")

        captured_count = 0
        rotation_angle = int(rotation) % 360

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[-] Failed to read frame from webcam.")
                break

            # Apply stream rotation if needed
            if rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            h, w, _ = frame.shape
            display_frame = frame.copy()
            faces = self.app.get(frame)

            # Draw detection bounding box
            if len(faces) > 0:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                bbox = face.bbox.astype(int)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
                cv2.putText(display_frame, "Face Detected", (bbox[0], max(20, bbox[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Top info bar
            cv2.rectangle(display_frame, (0, 0), (w, 60), (30, 30, 30), -1)
            cv2.putText(display_frame, f"ENROLLING: {name} | Captured: {captured_count} photos | Rot: {rotation_angle}°", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press [SPACE] capture | Press [r] rotate | Press [q] done", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            self._show_frame(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC to finish
                break

            elif key in (ord('r'), ord('R')):
                rotation_angle = (rotation_angle + 90) % 360
                print(f"[*] Rotated stream orientation to: {rotation_angle}°")

            elif key in (ord(' '), ord('c'), ord('C')):
                # Trigger capture
                if len(faces) == 0:
                    # Flash warning: No face detected
                    warn_frame = display_frame.copy()
                    cv2.rectangle(warn_frame, (50, h // 2 - 30), (w - 50, h // 2 + 30), (0, 0, 180), -1)
                    cv2.putText(warn_frame, "NO FACE DETECTED! Align face and try again.", (60, h // 2 + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    self._show_frame(window_name, warn_frame)
                    cv2.waitKey(800)
                    continue

                # Face is detected - Freeze frame and ask for verification [y/n]
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                bbox = face.bbox.astype(int)

                preview_frame = frame.copy()
                cv2.rectangle(preview_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)

                # Bottom dialog box for verification
                cv2.rectangle(preview_frame, (0, h - 80), (w, h), (20, 20, 20), -1)
                cv2.putText(preview_frame, "Accept this photo angle?", (15, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(preview_frame, "Press [y] = SAVE / ACCEPT   |   Press [n] = RETAKE / DISCARD", (15, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                self._show_frame(window_name, preview_frame)
                print(f"[*] Captured potential photo. Accept this angle for '{name}'? [y/n]")

                # Verification loop
                while True:
                    v_key = cv2.waitKey(0) & 0xFF
                    if v_key in (ord('y'), ord('Y'), ord(' ')):
                        # Accepted: Save image and embedding
                        captured_count += 1
                        timestamp = int(time.time() * 1000)
                        img_filename = f"{name.lower().replace(' ', '_')}_{captured_count}_{timestamp}.jpg"
                        img_save_path = os.path.join(person_folder, img_filename)
                        cv2.imwrite(img_save_path, frame)

                        # Extract and store normalized embedding
                        embedding = face.embedding / np.linalg.norm(face.embedding)
                        if name not in self.database:
                            self.database[name] = []
                        self.database[name].append(embedding)

                        print(f"    [+] [ACCEPTED] Saved photo #{captured_count} -> {img_save_path}")

                        # Visual confirmation
                        success_frame = preview_frame.copy()
                        cv2.rectangle(success_frame, (0, h - 80), (w, h), (0, 140, 0), -1)
                        cv2.putText(success_frame, f"SAVED photo #{captured_count}! Change angle and press [SPACE] for next.", (15, h - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                        self._show_frame(window_name, success_frame)
                        cv2.waitKey(600)
                        break

                    elif v_key in (ord('n'), ord('N')):
                        # Discarded
                        print("    [-] [RETAKE] Photo discarded. Re-align and press [SPACE] again.")
                        discard_frame = preview_frame.copy()
                        cv2.rectangle(discard_frame, (0, h - 80), (w, h), (0, 0, 180), -1)
                        cv2.putText(discard_frame, "Photo discarded. Re-align angle and press [SPACE] again.", (15, h - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                        self._show_frame(window_name, discard_frame)
                        cv2.waitKey(600)
                        break

                    elif v_key in (ord('q'), ord('Q'), 27):
                        print("[*] Exiting enrollment.")
                        break

        # Enrollment summary
        total_photos_now = len(self.database.get(name, []))
        if captured_count > 0:
            print(f"\n[+] Successfully added {captured_count} new photo(s) for '{name}'!")
            print(f"[+] Total photos now in database for '{name}': {total_photos_now}")
            print(f"[+] Live recognition resumed with updated model data!\n")

            confirm_frame = frame.copy()
            cv2.putText(confirm_frame, f"Registered '{name}' ({total_photos_now} total photos)!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self._show_frame(window_name, confirm_frame)
            cv2.waitKey(800)
            return True
        else:
            print(f"[*] No new photos were saved for '{name}'. Resuming recognition...\n")
            return False

    def recognize_frame(self, frame):
        """Run face detection and matching on a single frame/image."""
        faces = self.app.get(frame)
        detections = []

        for face in faces:
            query_emb = face.embedding / np.linalg.norm(face.embedding)
            best_name = "Unknown"
            best_score = 0.0

            # Compare against all enrolled identities
            for name, emb_list in self.database.items():
                for stored_emb in emb_list:
                    similarity = float(np.dot(stored_emb, query_emb))
                    if similarity > best_score:
                        best_score = similarity
                        if similarity >= self.threshold:
                            best_name = name

            bbox = face.bbox.astype(int)
            color = (0, 220, 0) if best_name != "Unknown" else (0, 0, 220)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            label = f"{best_name} ({best_score:.2f})"
            cv2.putText(frame, label, (bbox[0], max(20, bbox[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "name": best_name,
                "confidence": best_score,
                "bbox": bbox
            })

        return frame, detections

    def recognize_image_file(self, input_path: str, output_path: str = "output_result.jpg"):
        """Run recognition on an image file and save the annotated result."""
        img = cv2.imread(input_path)
        if img is None:
            print(f"[-] Could not read input image: {input_path}")
            return

        annotated_img, detections = self.recognize_frame(img)
        cv2.imwrite(output_path, annotated_img)
        print(f"[+] Recognition complete! Saved annotated image to: {output_path}")
        print("[*] Detected faces:")
        for idx, d in enumerate(detections, 1):
            print(f"    {idx}. Person: {d['name']} | Confidence: {d['confidence']:.2f}")

    def _show_frame(self, window_name: str, frame: np.ndarray, max_w: int = 1280, max_h: int = 720):
        """Helper to fit display frame proportionally to monitor screen bounds without distortion."""
        h, w = frame.shape[:2]
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale < 1.0:
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, resized)
        else:
            cv2.imshow(window_name, frame)

    def start_webcam(self, camera_source=0, rotation: int = 0):
        """
        Run real-time face recognition from a webcam or mobile IP camera feed.
        :param camera_source: Integer (0, 1 for webcam) or String (e.g. 'http://192.168.137.130:8080/' for Mobile IP Camera)
        :param rotation: Stream rotation angle (0, 90, 180, 270 degrees)
        """
        window_name = "InsightFace - Realtime Recognition"
        
        # Convert numeric string to int if needed
        if isinstance(camera_source, str):
            camera_source = camera_source.strip()
            if camera_source.isdigit():
                camera_source = int(camera_source)
            elif camera_source.startswith(("http://", "https://", "rtsp://")):
                # Auto-append '/video' for IP Webcam endpoints if missing
                url_clean = camera_source.rstrip('/')
                if not url_clean.endswith(('/video', '.mjpeg', '.mp4', '/shot.jpg')):
                    camera_source = url_clean + '/video'
                    print(f"[*] Auto-formatted stream URL for OpenCV -> {camera_source}")

        print(f"\n[*] Connecting to camera source: {camera_source}...")
        print("[*] Controls: Press 'e' or 'n' to Enroll  |  Press 'r' to Rotate Stream  |  Press 'q' to Exit.")
        
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            if isinstance(camera_source, int):
                print(f"[-] Cannot open webcam device {camera_source}. Scanning fallback devices...")
                for idx in [0, 1, 2, 3]:
                    if idx == camera_source:
                        continue
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        print(f"[+] Successfully connected to webcam device {idx}")
                        camera_source = idx
                        break
                else:
                    print("[-] No accessible webcam devices found.")
                    return
            else:
                print(f"[-] Could not connect to camera stream URL: '{camera_source}'")
                print("    Make sure your mobile phone and PC are on the same Wi-Fi network!")
                print("    Ensure IP Webcam server is STARTED on your mobile app.")
                return

        # Initialize resizable OpenCV window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        prev_time = time.time()
        fps = 0.0
        rotation_angle = int(rotation) % 360

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[-] Failed to capture frame from camera stream.")
                break

            # Apply stream rotation if requested
            if rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = curr_time

            annotated_frame, detections = self.recognize_frame(frame)

            h, w, _ = annotated_frame.shape

            # Draw top-right FPS & Rotation display
            fps_text = f"FPS: {fps:.1f} | Rot: {rotation_angle}°"
            cv2.putText(annotated_frame, fps_text, (w - 220, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Draw bottom control bar / guide
            cv2.rectangle(annotated_frame, (0, h - 30), (w, h), (30, 30, 30), -1)
            cv2.putText(annotated_frame, "Press [e] Enroll  |  Press [r] Rotate  |  Press [q] Quit", (15, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            self._show_frame(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key in (ord('r'), ord('R')):
                rotation_angle = (rotation_angle + 90) % 360
                print(f"[*] Rotated stream orientation to: {rotation_angle}°")

            elif key in (ord('e'), ord('E'), ord('n'), ord('N')):
                # Show pause overlay on video
                pause_frame = annotated_frame.copy()
                cv2.rectangle(pause_frame, (50, h // 2 - 35), (w - 50, h // 2 + 35), (0, 0, 0), -1)
                cv2.putText(pause_frame, "ENROLLMENT: Switch to terminal to enter name", (60, h // 2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                self._show_frame(window_name, pause_frame)
                cv2.waitKey(1)

                print("\n" + "=" * 50)
                print("--- Dynamic Face Enrollment ---")
                new_person_name = input("Enter person's name (or press Enter to cancel): ").strip()
                print("=" * 50)

                if new_person_name:
                    self.enroll_live_person(new_person_name, cap, window_name=window_name, rotation=rotation_angle)
                else:
                    print("[*] Enrollment cancelled. Resuming camera stream...")

        cap.release()
        cv2.destroyAllWindows()
        print("[*] Camera stream closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightFace Real-Time CPU Face Recognizer")
    parser.add_argument("--mode", choices=["webcam", "image"], help="Mode: 'webcam' or 'image'")
    parser.add_argument("--image", type=str, help="Path to test image file (for mode=image)")
    parser.add_argument("--camera", type=str, default="0", help="Webcam index (e.g. 0) or IP stream URL (e.g. http://192.168.137.130:8080/)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate stream angle: 0, 90, 180, or 270 degrees")
    args = parser.parse_args()

    recognizer = PersonRecognizer(known_faces_dir="known_faces")
    recognizer.load_known_faces()

    if args.mode == "webcam":
        recognizer.start_webcam(camera_source=args.camera, rotation=args.rotate)
    elif args.mode == "image":
        if args.image:
            recognizer.recognize_image_file(args.image)
        else:
            img_path = input("Enter path to test image (e.g. test.jpg): ").strip()
            if img_path:
                recognizer.recognize_image_file(img_path)
    else:
        print("\n" + "=" * 60)
        print("         InsightFace Real-Time Face Recognition Engine")
        print("=" * 60)
        print("Select Camera / Source:")
        print("  1. Default Desktop / Laptop Webcam (Index 0)")
        print("  2. Mobile IP Camera (e.g. http://192.168.137.130:8080/)")
        print("  3. Test on a Static Image File")
        print("  4. Exit")
        print("=" * 60)
        
        choice = input("Select option (1/2/3/4) or paste Mobile URL directly: ").strip()

        if choice == "1" or choice == "":
            recognizer.start_webcam(camera_source=0, rotation=args.rotate)
        elif choice == "2":
            url_in = input("\nEnter Mobile Camera URL (e.g. http://192.168.137.130:8080/): ").strip()
            if url_in:
                recognizer.start_webcam(camera_source=url_in, rotation=args.rotate)
            else:
                print("[!] No URL entered. Defaulting to webcam 0.")
                recognizer.start_webcam(camera_source=0, rotation=args.rotate)
        elif choice.startswith(("http://", "https://", "rtsp://")):
            # User pasted URL directly at main prompt
            recognizer.start_webcam(camera_source=choice, rotation=args.rotate)
        elif choice == "3":
            img_path = input("Enter path to test image (e.g. test.jpg): ").strip()
            if img_path:
                recognizer.recognize_image_file(img_path)
        else:
            print("Exiting.")
