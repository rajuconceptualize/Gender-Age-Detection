# Installation

gh repo clone rajuconceptualize/Gender-Age-Detection

# Create A Virtual environment
python -m venv <venv_name>

# activate the Virtual Environment
## MAC / Linux
source <venv_name>/bin/activate
## windows
<venv_name>\Scripts\activate

# Run the Script
python app.py

# Stop the Script
Hit CTRL + C 


## Top Backends for Face Recognition:

### RetinaFace (Highly Recommended):

**Accuracy:** Very high, including for diverse gender and ethnic groups.

**Speed:** Moderate to high.

**Notes:** RetinaFace is one of the most advanced face detection models and is well-suited for recognizing both male and female faces with high precision.


###  MTCNN (Multi-Task Cascaded Convolutional Networks):

**Accuracy:** High.

**Speed:** Moderate.

**Notes:** MTCNN is robust in face detection and alignment and works well across genders. It’s a solid choice if you want balanced speed and accuracy.



### Dlib:

**Accuracy:** High, especially in well-lit and frontal faces.

**Speed:** Slower compared to RetinaFace and MTCNN.

**Notes:**  Works well for gender-specific face recognition, though it can struggle with side profiles or partially obscured faces.


### Mediapipe:

**Accuracy:** Moderate to high.

**Speed:** Very fast.

**Notes:**  Known for its high speed, Mediapipe is accurate and lightweight, making it a good option for real-time applications.

## Backends to Consider for Specific Use Cases:

### SSD (Single Shot Multibox Detector):

**Accuracy:** Moderate.

**Speed:** Fast.

**Notes:**  Great for real-time applications but less accurate than RetinaFace or MTCNN.

### Yolov8:

**Accuracy:** Moderate to high.

**Speed:** Fast.

**Notes:**  YOLOv8 is the latest version of the YOLO family, optimized for real-time applications but with strong accuracy.


