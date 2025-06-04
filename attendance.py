# import streamlit as st
# import cv2
# import os
# import numpy as np
# import csv
# import pandas as pd
# import time
# from datetime import datetime
# from PIL import Image

# def initialize():
#     os.makedirs("Training_Images", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)
#     os.makedirs("Training_Image_Label", exist_ok=True)
#     os.makedirs("SmartAttendance", exist_ok=True)
#     os.makedirs("ManualAttendance", exist_ok=True)

# def get_haarcascade_path():
#     """Returns the correct absolute path for the Haarcascade file."""
#     cascade_path = os.path.join(os.getcwd(), "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/haarcascade_frontalface_default.xml")
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError(f"Error: Haarcascade file not found at '{cascade_path}'. Please check the file location.")
#     return cascade_path

# def capture_images(name, enrollment):
#     cascade_path = get_haarcascade_path()
#     detector = cv2.CascadeClassifier(cascade_path)

#     if detector.empty():
#         raise RuntimeError("Error: Haar cascade classifier failed to load. Check OpenCV installation and XML file.")

#     cap = cv2.VideoCapture(0)
#     count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             count += 1
#             image_path = f"Training_Images/{name}.{enrollment}.{count}.jpg"
#             cv2.imwrite(image_path, gray[y:y+h, x:x+w])
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             if count >= 2:
#                 update_student_details_csv(enrollment, name)
#                 st.success("Images Captured and Details Saved Successfully!")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cv2.imshow("Capture Images", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def update_student_details_csv(enrollment, name):
#     csv_file_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"
#     with open(csv_file_path, mode="a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([enrollment, name, time.strftime("%Y-%m-%d %H:%M:%S")])

# def train_images():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels("Training_Images")
    
#     if len(faces) == 0 or len(ids) == 0:
#         st.error("No training images found. Capture images first!")
#         return

#     recognizer.train(faces, np.array(ids))
#     recognizer.save("C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/Training_Image_Label/trainner.ymll")
#     st.success("Model Trained Successfully!")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces, ids = [], []
#     for image_path in image_paths:
#         img = Image.open(image_path).convert("L")
#         img_np = np.array(img, "uint8")
#         id = int(os.path.split(image_path)[-1].split(".")[1])
#         faces.append(img_np)
#         ids.append(id)
#     return faces, ids

# def mark_attendance(student_id, subject):
#     filename = f"SmartAttendance/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
#     with open(filename, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([student_id, time.strftime("%Y-%m-%d %H:%M:%S")])
#     st.success(f"Attendance marked for ID: {student_id}.")

# def smart_attendance(subject):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer_path = "Training_Image_Label/trainner.yml"

#     if not os.path.exists(recognizer_path):
#         st.error("Error: Training model not found. Train the model first!")
#         return

#     recognizer.read(recognizer_path)

#     cascade_path = get_haarcascade_path()
#     detector = cv2.CascadeClassifier(cascade_path)

#     if detector.empty():
#         raise RuntimeError("Error: Haar cascade classifier failed to load.")

#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Error: Could not read frame from camera.")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#             if confidence < 100:
#                 mark_attendance(id, subject)
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cv2.imshow("Smart Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     initialize()
#     st.title("AI Attendance System")
#     menu = ["Home", "Capture Images", "Train Model", "Smart Attendance", "Manual Attendance", "Admin Panel"]
#     choice = st.sidebar.selectbox("Navigation", menu)

#     if choice == "Capture Images":
#         name = st.text_input("Enter Student Name")
#         enrollment = st.text_input("Enter Enrollment Number")
#         if st.button("Capture Images"):
#             if name and enrollment.isdigit():
#                 capture_images(name, enrollment)
#             else:
#                 st.error("Invalid Name or Enrollment Number")

#     elif choice == "Train Model":
#         if st.button("Train Model"):
#             train_images()

#     elif choice == "Smart Attendance":
#         subject = st.text_input("Enter Subject Name")
#         if st.button("Start Attendance"):
#             smart_attendance(subject)

#     elif choice == "Manual Attendance":
#         enrollment = st.text_input("Enter Enrollment Number")
#         name = st.text_input("Enter Name")
#         if st.button("Submit Attendance"):
#             update_student_details_csv(enrollment, name)
#             st.success("Manual Attendance Marked Successfully!")

#     elif choice == "Admin Panel":
#         st.subheader("Registered Students")
#         try:
#             student_data = pd.read_csv("C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv")
#             st.dataframe(student_data)
#         except FileNotFoundError:
#             st.error("No registered students found.")

# if __name__ == "__main__":
#     main()

# # ---------------------------------------------------------------MAIN CODE IS UP 



# import streamlit as st
# import cv2
# import os
# import numpy as np
# import csv
# import pandas as pd
# import time
# from datetime import datetime
# from PIL import Image

# def initialize():
#     os.makedirs("Training_Images", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)
#     os.makedirs("Training_Image_Label", exist_ok=True)
#     os.makedirs("SmartAttendance", exist_ok=True)
#     os.makedirs("ManualAttendance", exist_ok=True)

# def get_haarcascade_path():
#     cascade_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/haarcascade_frontalface_default.xml"
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError("Error: Haarcascade file not found.")
#     return cascade_path

# def capture_images(name, enrollment):
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
#     count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             count += 1
#             image_path = f"Training_Images/{name}.{enrollment}.{count}.jpg"
#             cv2.imwrite(image_path, gray[y:y+h, x:x+w])
#             if count >= 5:
#                 update_student_details_csv(enrollment, name)
#                 st.success("Images Captured Successfully!")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cv2.imshow("Capture Images", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def update_student_details_csv(enrollment, name):
#     csv_file_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"
#     with open(csv_file_path, "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([enrollment, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# def train_images():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels("Training_Images")
    
#     if not faces:
#         st.error("No training images found!")
#         return

#     recognizer.train(faces, np.array(ids))
#     recognizer.save("C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/Training_Image_Label/trainner.ymll")
#     st.success("Model Trained Successfully!")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces, ids = [], []
#     for image_path in image_paths:
#         img = Image.open(image_path).convert("L")
#         img_np = np.array(img, "uint8")
#         id = int(os.path.split(image_path)[-1].split(".")[1])
#         faces.append(img_np)
#         ids.append(id)
#     return faces, ids

# def mark_attendance(student_id, subject, image_path):
#     filename = f"SmartAttendance/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
#     with open(filename, "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path])
#     st.success(f"Attendance marked for ID: {student_id}.")

# def smart_attendance(subject):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/Training_Image_Label/trainner.ymll")
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#             if confidence < 100:
#                 image_path = f"SmartAttendance/{subject}_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 cv2.imwrite(image_path, frame)
#                 mark_attendance(id, subject, image_path)
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cv2.imshow("Smart Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def manual_attendance(enrollment, name):
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     if ret:
#         image_path = f"ManualAttendance/{enrollment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#         cv2.imwrite(image_path, frame)
#         update_student_details_csv(enrollment, name)
#         st.success("Manual Attendance Marked Successfully!")
#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     initialize()
#     st.title("Face Recognition Attendance System")
#     menu = ["Capture Images", "Train Model", "Smart Attendance", "Manual Attendance", "Admin Panel"]
#     choice = st.sidebar.selectbox("Navigation", menu)

#     if choice == "Capture Images":
#         name = st.text_input("Enter Student Name")
#         enrollment = st.text_input("Enter Enrollment Number")
#         if st.button("Capture"):
#             capture_images(name, enrollment)

#     elif choice == "Train Model":
#         if st.button("Train"):
#             train_images()

#     elif choice == "Smart Attendance":
#         subject = st.text_input("Enter Subject")
#         if st.button("Start"):
#             smart_attendance(subject)

#     elif choice == "Manual Attendance":
#         enrollment = st.text_input("Enter Enrollment Number")
#         name = st.text_input("Enter Name")
#         if st.button("Submit Attendance"):
#             manual_attendance(enrollment, name)

#     elif choice == "Admin Panel":
#         st.subheader("Student Records")
#         try:
#             student_data = pd.read_csv("C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv")
#             st.dataframe(student_data)
#         except FileNotFoundError:
#             st.error("No records found.")

# if __name__ == "__main__":
#     main()





# import streamlit as st
# import cv2
# import os
# import numpy as np
# import csv
# import pandas as pd
# import time
# from datetime import datetime
# from PIL import Image

# def initialize():
#     os.makedirs("Training_Images", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)
#     os.makedirs("Training_Image_Label", exist_ok=True)
#     os.makedirs("SmartAttendance", exist_ok=True)
#     os.makedirs("ManualAttendance", exist_ok=True)

# def get_haarcascade_path():
#     cascade_path = "haarcascade_frontalface_default.xml"
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError(f"Error: Haarcascade file not found at '{cascade_path}'.")
#     return cascade_path

# def capture_images(name, enrollment):
#     cascade_path = get_haarcascade_path()
#     detector = cv2.CascadeClassifier(cascade_path)
#     cap = cv2.VideoCapture(0)
#     count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             count += 1
#             image_path = f"Training_Images/{name}.{enrollment}.{count}.jpg"
#             cv2.imwrite(image_path, gray[y:y+h, x:x+w])
#             if count >= 5:
#                 update_student_details_csv(enrollment, name)
#                 st.success("Images Captured Successfully!")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cv2.imshow("Capture Images", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def update_student_details_csv(enrollment, name):
#     csv_file_path = "StudentDetails/StudentDetails.csv"
#     with open(csv_file_path, "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([enrollment, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# def train_images():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels("Training_Images")
    
#     if not faces:
#         st.error("No training images found!")
#         return

#     recognizer.train(faces, np.array(ids))
#     recognizer.save("Training_Image_Label/trainner.yml")
#     st.success("Model Trained Successfully!")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces, ids = [], []
#     for image_path in image_paths:
#         img = Image.open(image_path).convert("L")
#         img_np = np.array(img, "uint8")
#         id = int(os.path.split(image_path)[-1].split(".")[1])
#         faces.append(img_np)
#         ids.append(id)
#     return faces, ids

# def mark_attendance(student_id, subject, image_path, mode):
#     filename = f"{mode}Attendance/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
#     with open(filename, "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path])
#     st.success(f"{mode} Attendance marked for ID: {student_id}.")

# def smart_attendance(subject):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer_path = "Training_Image_Label/trainner.yml"
    
#     if not os.path.exists(recognizer_path):
#         st.error("Model not trained. Train first!")
#         return
    
#     recognizer.read(recognizer_path)
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Camera error.")
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#             if confidence < 100:
#                 image_path = f"SmartAttendance/{subject}_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 cv2.imwrite(image_path, frame)
#                 mark_attendance(id, subject, image_path, "Smart")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cv2.imshow("Smart Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def manual_attendance(enrollment, name, subject):
#     image_path = f"ManualAttendance/{subject}_{enrollment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     if ret:
#         cv2.imwrite(image_path, frame)
#         mark_attendance(enrollment, subject, image_path, "Manual")
#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     initialize()
#     st.title("AI Attendance System")
#     menu = ["Capture Images", "Train Model", "Smart Attendance", "Manual Attendance", "Admin Panel"]
#     choice = st.sidebar.selectbox("Navigation", menu)

#     if choice == "Capture Images":
#         name = st.text_input("Enter Student Name")
#         enrollment = st.text_input("Enter Enrollment Number")
#         if st.button("Capture"):
#             capture_images(name, enrollment)

#     elif choice == "Train Model":
#         if st.button("Train"):
#             train_images()

#     elif choice == "Smart Attendance":
#         subject = st.text_input("Enter Subject")
#         if st.button("Start"):
#             smart_attendance(subject)

#     elif choice == "Manual Attendance":
#         enrollment = st.text_input("Enter Enrollment Number")
#         name = st.text_input("Enter Name")
#         subject = st.text_input("Enter Subject")
#         if st.button("Submit"):
#             manual_attendance(enrollment, name, subject)

#     elif choice == "Admin Panel":
#         st.subheader("Student Records")
#         try:
#             student_data = pd.read_csv("StudentDetails/StudentDetails.csv")
#             st.dataframe(student_data)
#         except FileNotFoundError:
#             st.error("No records found.")

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import cv2
# import os
# import numpy as np
# import csv
# import pandas as pd
# import time
# from datetime import datetime
# from PIL import Image

# def initialize():
#     os.makedirs("Training_Images", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)
#     os.makedirs("Training_Image_Label", exist_ok=True)
#     os.makedirs("SmartAttendance", exist_ok=True)
#     os.makedirs("ManualAttendance", exist_ok=True)

# def get_haarcascade_path():
#     cascade_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/haarcascade_frontalface_default.xml"
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError("Error: Haarcascade file not found!")
#     return cascade_path

# def capture_images(name, enrollment):
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
#     count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
#         for (x, y, w, h) in faces:
#             count += 1
#             image_path = f"Training_Images/{name}.{enrollment}.{count}.jpg"
#             cv2.imwrite(image_path, gray[y:y+h, x:x+w])
            
#             if count >= 5:
#                 update_student_details_csv(enrollment, name)
#                 st.success("Images Captured Successfully!")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return
        
#         cv2.imshow("Capture Images", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# def update_student_details_csv(enrollment, name):
#     csv_file_path = "Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"
#     with open(csv_file_path, "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([enrollment, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# def train_images():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels("Training_Images")
    
#     if not faces:
#         st.error("No training images found! Capture images first.")
#         return
    
#     recognizer.train(faces, np.array(ids))
#     recognizer.save("Training_Image_Label/trainner.yml")
#     st.success("Model Trained Successfully!")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces, ids = [], []
#     for image_path in image_paths:
#         img = Image.open(image_path).convert("L")
#         img_np = np.array(img, "uint8")
#         id = int(os.path.split(image_path)[-1].split(".")[1])
#         faces.append(img_np)
#         ids.append(id)
#     return faces, ids

# def mark_attendance(student_id, subject, image_path, mode="SmartAttendance"):
#     filename = f"{mode}/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
#     with open(filename, "a", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow([student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path])
#     st.success(f"Attendance marked for ID: {student_id}.")

# def smart_attendance(subject):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("Training_Image_Label/trainner.yml")
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Camera error.")
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#             if confidence < 100:
#                 image_path = f"SmartAttendance/{subject}_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 cv2.imwrite(image_path, frame)
#                 mark_attendance(id, subject, image_path, "SmartAttendance")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return
        
#         cv2.imshow("Smart Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# def manual_attendance(enrollment, name, subject):
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     if not ret:
#         st.error("Camera not working!")
#         return
    
#     image_path = f"ManualAttendance/{subject}_{enrollment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#     cv2.imwrite(image_path, frame)
#     mark_attendance(enrollment, subject, image_path, "ManualAttendance")
#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     initialize()
#     st.title("Face Recognition Attendance System")
#     menu = ["Home", "Capture Images", "Train Model", "Smart Attendance", "Manual Attendance", "Admin Panel"]
#     choice = st.sidebar.selectbox("Navigation", menu)

#     if choice == "Home":
#         st.write("Welcome to the Face Recognition Based Attendance System!")

#     elif choice == "Capture Images":
#         name = st.text_input("Enter Student Name")
#         enrollment = st.text_input("Enter Enrollment Number")
#         if st.button("Capture"):
#             capture_images(name, enrollment)

#     elif choice == "Train Model":
#         if st.button("Train"):
#             train_images()

#     elif choice == "Smart Attendance":
#         subject = st.text_input("Enter Subject")
#         if st.button("Start"):
#             smart_attendance(subject)

#     elif choice == "Manual Attendance":
#         enrollment = st.text_input("Enter Enrollment Number")
#         name = st.text_input("Enter Name")
#         subject = st.text_input("Enter Subject")
#         if st.button("Mark Attendance"):
#             manual_attendance(enrollment, name, subject)

#     elif choice == "Admin Panel":
#         try:
#             student_data = pd.read_csv("Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv")
#             st.dataframe(student_data)
#         except FileNotFoundError:
#             st.error("No student data found.")

# if __name__ == "__main__":
#     main()
# ---------------------------------------------------------- MAIN CODE 2



# import streamlit as st
# import cv2
# import os
# import numpy as np
# import csv
# import pandas as pd
# from datetime import datetime
# from PIL import Image

# def initialize():
#     os.makedirs("Training_Images", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)
#     os.makedirs("Training_Image_Label", exist_ok=True)
#     os.makedirs("SmartAttendance", exist_ok=True)
#     os.makedirs("ManualAttendance", exist_ok=True)

# def get_haarcascade_path():
#     cascade_path = "Attendance-Management-System-using-Face-Recognition-master/haarcascade_frontalface_default.xml"
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError("Haarcascade file not found!")
#     return cascade_path

# def update_student_details_csv(enrollment, name):
#     csv_file_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"
#     headers = ["Enrollment", "Name", "Timestamp"]
#     file_exists = os.path.exists(csv_file_path)
#     with open(csv_file_path, "a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(headers)
#         writer.writerow([enrollment, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# def capture_images(name, enrollment):
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
#     count = 0
#     while count < 5:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
#         for (x, y, w, h) in faces:
#             count += 1
#             cv2.imwrite(f"Training_Images/{name}.{enrollment}.{count}.jpg", gray[y:y+h, x:x+w])
#         cv2.imshow("Capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     if count >= 5:
#         update_student_details_csv(enrollment, name)
#         st.success("Images Captured Successfully!")

# def train_images():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels("Training_Images")
#     if not faces:
#         st.error("No images found!")
#         return
#     recognizer.train(faces, np.array(ids))
#     recognizer.save("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
#     st.success("Images Trained Successfully!")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces, ids = [], []
#     for image_path in image_paths:
#         img = Image.open(image_path).convert("L")
#         img_np = np.array(img, "uint8")
#         id = int(os.path.split(image_path)[-1].split(".")[1])
#         faces.append(img_np)
#         ids.append(id)
#     return faces, ids

# def mark_attendance(student_id, subject, image_path, method):
#     folder = "SmartAttendance" if method == "Smart" else "ManualAttendance"
#     filename = f"{folder}/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
#     headers = ["Student ID", "Timestamp", "Image Path"]
#     file_exists = os.path.exists(filename)
#     with open(filename, "a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(headers)
#         writer.writerow([student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path])
#     st.success(f"{method} Attendance Marked for ID: {student_id}.")

# def smart_attendance(subject):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#             if confidence < 100:
#                 image_path = f"SmartAttendance/{subject}_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 cv2.imwrite(image_path, frame)
#                 mark_attendance(id, subject, image_path, "Smart")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return
#         cv2.imshow("Smart Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# def manual_attendance():
#     name = st.text_input("Enter Student Name")
#     student_id = st.text_input("Enter Student ID")
#     subject = st.text_input("Enter Subject")
#     if st.button("Mark Attendance"):
#         image_path = f"ManualAttendance/{subject}_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#         mark_attendance(student_id, subject, image_path, "Manual")

# def main():
#     initialize()
#     st.title("Face Recognition Attendance System")
#     menu = ["Home", "Capture Images", "Train Images", "Smart Attendance", "Manual Attendance", "Admin Panel"]
#     choice = st.sidebar.selectbox("Navigation", menu)
#     if choice == "Home":
#         st.write("Welcome to the Face Recognition Attendance System!")
#     elif choice == "Capture Images":
#         name = st.text_input("Enter Student Name")
#         enrollment = st.text_input("Enter Enrollment Number")
#         if st.button("Capture"):
#             capture_images(name, enrollment)
#     elif choice == "Train Images":
#         if st.button("Train"):
#             train_images()
#     elif choice == "Smart Attendance":
#         subject = st.text_input("Enter Subject")
#         if st.button("Start"):
#             smart_attendance(subject)
#     elif choice == "Manual Attendance":
#         manual_attendance()
#     elif choice == "Admin Panel":
#         st.subheader("Student Records")
#         try:
#             student_data = pd.read_csv("C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv", on_bad_lines='skip')
#             st.dataframe(student_data)
#         except FileNotFoundError:
#             st.error("No records found.")

# if __name__ == "__main__":
#     main()



# abve as main code 




            
            
# import streamlit as st
# import cv2
# import os
# import numpy as np
# import csv
# import pandas as pd
# from datetime import datetime
# from PIL import Image

# def initialize():
#     os.makedirs("Training_Images", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)
#     os.makedirs("Training_Image_Label", exist_ok=True)
#     os.makedirs("SmartAttendance", exist_ok=True)
#     os.makedirs("ManualAttendance", exist_ok=True)

# def get_haarcascade_path():
#     cascade_path = "Attendance-Management-System-using-Face-Recognition-master/haarcascade_frontalface_default.xml"
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError("Haarcascade file not found!")
#     return cascade_path

# def update_student_details_csv(enrollment, name):
#     csv_file_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"
#     headers = ["Enrollment", "Name", "Timestamp"]
#     file_exists = os.path.exists(csv_file_path)
#     with open(csv_file_path, "a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(headers)
#         writer.writerow([enrollment, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# def capture_images(name, enrollment):
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
#     count = 0
#     while count < 5:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
#         for (x, y, w, h) in faces:
#             count += 1
#             cv2.imwrite(f"Training_Images/{name}.{enrollment}.{count}.jpg", gray[y:y+h, x:x+w])
#         cv2.imshow("Capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     if count >= 5:
#         update_student_details_csv(enrollment, name)
#         st.success("Images Captured Successfully!")

# def train_images():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels("Training_Images")
#     if not faces:
#         st.error("No images found!")
#         return
#     recognizer.train(faces, np.array(ids))
#     recognizer.save("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
#     st.success("Images Trained Successfully!")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces, ids = [], []
#     for image_path in image_paths:
#         img = Image.open(image_path).convert("L")
#         img_np = np.array(img, "uint8")
#         id = int(os.path.split(image_path)[-1].split(".")[1])
#         faces.append(img_np)
#         ids.append(id)
#     return faces, ids

# def mark_attendance(student_id, subject, image_path, method):
#     folder = "SmartAttendance" if method == "Smart" else "ManualAttendance"
#     filename = f"{folder}/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
#     headers = ["Student ID", "Timestamp", "Image Path"]
#     file_exists = os.path.exists(filename)
#     with open(filename, "a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(headers)
#         writer.writerow([student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path])
#     st.success(f"{method} Attendance Marked for ID: {student_id}.")

# def smart_attendance(subject):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#             if confidence < 100:
#                 image_path = f"SmartAttendance/{subject}_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 cv2.imwrite(image_path, frame)
#                 mark_attendance(id, subject, image_path, "Smart")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return
#         cv2.imshow("Smart Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# def manual_attendance():
#     name = st.text_input("Enter Student Name")
#     student_id = st.text_input("Enter Student ID")
#     subject = st.text_input("Enter Subject")
#     if st.button("Mark Attendance"):
#         cap = cv2.VideoCapture(0)
#         ret, frame = cap.read()
#         cap.release()
#         if ret:
#             image_path = f"ManualAttendance/{subject}_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             cv2.imwrite(image_path, frame)
#             mark_attendance(student_id, subject, image_path, "Manual")
#         else:
#             st.error("Failed to capture image.")
# def main():
#     initialize()
#     st.title("Face Recognition Attendance System")
#     menu = ["Home", "Capture Images", "Train Images", "Smart Attendance", "Manual Attendance", "Admin Panel"]
#     choice = st.sidebar.selectbox("Navigation", menu)
#     if choice == "Home":
#         st.write("Welcome to the Face Recognition Attendance System!")
#     elif choice == "Capture Images":
#         name = st.text_input("Enter Student Name")
#         enrollment = st.text_input("Enter Enrollment Number")
#         if st.button("Capture"):
#             capture_images(name, enrollment)
#     elif choice == "Train Images":
#         if st.button("Train"):
#             train_images()
#     elif choice == "Smart Attendance":
#         subject = st.text_input("Enter Subject")
#         if st.button("Start"):
#             smart_attendance(subject)
#     elif choice == "Manual Attendance":
#         manual_attendance()
#     elif choice == "Admin Panel":
#         st.subheader("Student Records")
#         try:
#             student_data = pd.read_csv("C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv", on_bad_lines='skip')
#             st.dataframe(student_data)
#         except FileNotFoundError:
#             st.error("No records found.")

# if __name__ == "__main__":
#     main()            




# import streamlit as st
# import cv2
# import os
# import numpy as np
# import csv
# import pandas as pd
# from datetime import datetime
# from PIL import Image

# def initialize():
#     os.makedirs("Training_Images", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)
#     os.makedirs("Training_Image_Label", exist_ok=True)
#     os.makedirs("SmartAttendance", exist_ok=True)
#     os.makedirs("ManualAttendance", exist_ok=True)

# def get_haarcascade_path():
#     cascade_path = "Attendance-Management-System-using-Face-Recognition-master/haarcascade_frontalface_default.xml"
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError("Haarcascade file not found!")
#     return cascade_path

# def update_student_details_csv(enrollment, name):
#     csv_file_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"
#     headers = ["Enrollment", "Name", "Timestamp"]
#     file_exists = os.path.exists(csv_file_path)

#     with open(csv_file_path, "a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(headers)
#         writer.writerow([enrollment, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# def capture_images(name, enrollment):
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)
#     count = 0

#     while count < 5:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             count += 1
#             cv2.imwrite(f"Training_Images/{name}.{enrollment}.{count}.jpg", gray[y:y+h, x:x+w])

#         cv2.imshow("Capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     if count >= 5:
#         update_student_details_csv(enrollment, name)
#         st.success("Images Captured Successfully!")

# def train_images():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels("Training_Images")

#     if not faces:
#         st.error("No images found!")
#         return

#     recognizer.train(faces, np.array(ids))
#     recognizer.save("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
#     st.success("Images Trained Successfully!")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
#     faces, ids = [], []

#     for image_path in image_paths:
#         try:
#             img = Image.open(image_path).convert("L")
#             img_np = np.array(img, "uint8")
#             id = int(os.path.split(image_path)[-1].split(".")[1])
#             faces.append(img_np)
#             ids.append(id)
#         except Exception as e:
#             print(f"Skipping {image_path}: {e}")

#     return faces, ids

# def mark_attendance(student_id, subject, image_path, method):
#     folder = "SmartAttendance" if method == "Smart" else "ManualAttendance"
#     filename = f"{folder}/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
#     headers = ["Student ID", "Timestamp", "Image Path"]
#     file_exists = os.path.exists(filename)

#     with open(filename, "a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(headers)
#         writer.writerow([student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path])

#     st.success(f"{method} Attendance Marked for ID: {student_id}.")

# def smart_attendance(subject):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
#     detector = cv2.CascadeClassifier(get_haarcascade_path())
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#             if confidence < 100:
#                 image_path = f"SmartAttendance/{subject}_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 cv2.imwrite(image_path, frame)
#                 mark_attendance(id, subject, image_path, "Smart")
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return

#         cv2.imshow("Smart Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def manual_attendance():
#     name = st.text_input("Enter Student Name")
#     student_id = st.text_input("Enter Student ID")
#     subject = st.text_input("Enter Subject")

#     if st.button("Mark Attendance"):
#         cap = cv2.VideoCapture(0)
#         ret, frame = cap.read()
#         cap.release()

#         if ret:
#             image_path = f"ManualAttendance/{subject}_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#             cv2.imwrite(image_path, frame)
#             mark_attendance(student_id, subject, image_path, "Manual")
#         else:
#             st.error("Failed to capture image.")

# def main():
#     initialize()
#     st.title("Face Recognition Attendance System")
#     menu = ["Home", "Capture Images", "Train Images", "Smart Attendance", "Manual Attendance", "Admin Panel"]
#     choice = st.sidebar.selectbox("Navigation", menu)

#     if choice == "Home":
#         st.write("Welcome to the Face Recognition Attendance System!")
#     elif choice == "Capture Images":
#         name = st.text_input("Enter Student Name")
#         enrollment = st.text_input("Enter Enrollment Number")
#         if st.button("Capture"):
#             capture_images(name, enrollment)
#     elif choice == "Train Images":
#         if st.button("Train"):
#             train_images()
#     elif choice == "Smart Attendance":
#         subject = st.text_input("Enter Subject")
#         if st.button("Start"):
#             smart_attendance(subject)
#     elif choice == "Manual Attendance":
#         manual_attendance()
#     elif choice == "Admin Panel":
#         st.subheader("Student Records")
#         csv_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"

#         try:
#             student_data = pd.read_csv(csv_path, on_bad_lines="skip")  # Skips invalid rows
#             st.dataframe(student_data)
#         except pd.errors.ParserError:
#             st.error("Error reading the CSV file! Some lines might be corrupted.")
#         except FileNotFoundError:
#             st.error("No records found.")

# if __name__ == "__main__":
#     main()

# the above code is no error


import streamlit as st
import cv2
import os
import numpy as np
import csv
import pandas as pd
from datetime import datetime
from PIL import Image

# Directory Initialization
def initialize():
    os.makedirs("Training_Images", exist_ok=True)
    os.makedirs("StudentDetails", exist_ok=True)
    os.makedirs("Training_Image_Label", exist_ok=True)
    os.makedirs("SmartAttendance", exist_ok=True)
    os.makedirs("ManualAttendance", exist_ok=True)

# Path for Haarcascade File
def get_haarcascade_path():
    cascade_path = "Attendance-Management-System-using-Face-Recognition-master/haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError("Haarcascade file not found!")
    return cascade_path

# Save Student Details
def update_student_csv(enrollment, name):
    csv_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"
    
    headers = ["Enrollment", "Name", "Timestamp"]
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([enrollment, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Capture Student Images
def capture_images(name, enrollment):
    detector = cv2.CascadeClassifier(get_haarcascade_path())
    cap = cv2.VideoCapture(0)
    count = 0

    while count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"Training_Images/{name}.{enrollment}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count >= 5:
        update_student_csv(enrollment, name)
        st.success("Images Captured Successfully!")

# Train Images
def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels("Training_Images")

    if not faces:
        st.error("No images found!")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
    st.success("Images Trained Successfully!")

# Get Images and Labels for Training
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, ids = [], []

    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert("L")
            img_np = np.array(img, "uint8")
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(img_np)
            ids.append(id)
        except:
            pass

    return faces, ids

# Mark Attendance
def mark_attendance(student_id, subject, image_path, method):
    folder = "SmartAttendance" if method == "Smart" else "ManualAttendance"
    filename = f"{folder}/{subject}_{datetime.now().strftime('%Y-%m')}.csv"
    headers = ["Student ID", "Timestamp", "Image Path"]
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path])

    st.success(f"{method} Attendance Marked for ID: {student_id}.")

# Smart Attendance Using Face Recognition
def smart_attendance(subject):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("C:/Users/sivas/OneDrive/Desktop/final project/Training_Image_Label/trainner.yml")
    detector = cv2.CascadeClassifier(get_haarcascade_path())
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 100:
                image_path = f"SmartAttendance/{subject}_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(image_path, frame)
                mark_attendance(id, subject, image_path, "Smart")
                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Manual Attendance
def manual_attendance():
    name = st.text_input("Enter Student Name")
    student_id = st.text_input("Enter Student ID")
    subject = st.text_input("Enter Subject")

    if st.button("Mark Attendance"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            image_path = f"ManualAttendance/{subject}_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(image_path, frame)
            mark_attendance(student_id, subject, image_path, "Manual")
        else:
            st.error("Failed to capture image.")

# Streamlit App
def main():
    initialize()
    st.title("Face Recognition Attendance System")
    menu = ["Home", "Capture Images", "Train Images", "Smart Attendance", "Manual Attendance", "Admin Panel"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.write("Welcome to the Face Recognition Attendance System!")
    elif choice == "Capture Images":
        name = st.text_input("Enter Student Name")
        enrollment = st.text_input("Enter Enrollment Number")
        if st.button("Capture"):
            capture_images(name, enrollment)
    elif choice == "Train Images":
        if st.button("Train"):
            train_images()
    elif choice == "Smart Attendance":
        subject = st.text_input("Enter Subject")
        if st.button("Start"):
            smart_attendance(subject)
    elif choice == "Manual Attendance":
        manual_attendance()
    elif choice == "Admin Panel":
        st.subheader("Student Records")
        csv_path = "C:/Users/sivas/OneDrive/Desktop/final project/Attendance-Management-System-using-Face-Recognition-master/StudentDetails/StudentDetails.csv"

        try:
            student_data = pd.read_csv(csv_path, on_bad_lines="skip")
            st.dataframe(student_data)
        except:
            st.error("No records found.")

if __name__ == "__main__":
    main()


# it is above code is final




