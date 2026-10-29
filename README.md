VISIONGUARD
Real-Time Driver Drowsiness and Fatigue Detection Using Multimodal Facial Analysis
VISIONGUARD is an academic computer-vision project that aims to detect visible signs of driver drowsiness using a webcam. The system analyzes facial landmarks, eye closure, and head position, then displays the driver's status on a web dashboard and can trigger an audio warning.
Project status: Academic prototype. This README describes the proposed system from the project presentation. Add and test the implementation before describing it as fully functional. It is not a certified driver-safety system.

Overview
Driver fatigue can reduce attention, concentration, and reaction time. VISIONGUARD proposes a low-cost camera-based approach to monitor facial signs that may indicate drowsiness, including prolonged eye closure, yawning, and head movement.
Objectives
- Detect the driver's face using a webcam.
- Monitor eye closure using the Eye Aspect Ratio (EAR).
- Analyze facial landmarks for possible yawning.
- Monitor head position and tilt.
- Classify the driver as Alert or Drowsy.
- Display the results on a web dashboard.
- Generate an audio alarm when drowsiness is detected.
System Pipeline
Webcam
  ↓
MediaPipe Face Landmarker
  ↓
Facial Landmark Detection
  ↓
Eye + Mouth + Head Analysis
  ↓
EAR + Facial Position Logic + MAR
  ↓
Drowsiness Decision
  ↓
Web Dashboard + Audio Alarm
Working Principle
1. Camera capture: The webcam captures the driver's face.
2. Facial landmark detection: MediaPipe Face Landmarker identifies points around the eyes, mouth, and face.
3. Eye analysis: EAR is calculated to estimate whether the eyes are open or closed.
4. Head-position analysis: Facial landmarks are used to identify head position, such as normal, tilted, or down.
5. Drowsiness detection: Eye closure and head position are analyzed over time. If configured conditions persist beyond a threshold, the system flags possible drowsiness.
6. Alert: The dashboard displays a warning and the audio alarm can be triggered.
Technologies
Area	Technologies
Computer Vision	MediaPipe Face Landmarker 1.1.0, OpenCV
Analysis	Eye Aspect Ratio (EAR), facial landmarks, head-position logic
Backend	Python, Flask
Libraries	NumPy, SciPy, imutils
Frontend	HTML, CSS, JavaScript
Alert	WAV audio alarm
Development	VS Code, Python virtual environment, webcam, web browser


Expected Results
- Real-time monitoring using a normal webcam.
- Detection of prolonged eye closure.
- Head-position / movement monitoring.
- Real-time driver-status display.
- Audible warning when drowsiness conditions are detected.
- A simple, low-cost prototype and web interface.
Future Enhancements
- Add head-nodding and yawning detection.
- Improve detection using machine-learning models.
- Add driver-specific calibration.
- Store monitoring-session statistics.
- Add mobile or IoT integration.
- Improve performance under low-light conditions.
- Explore emergency-notification features.
Limitations and Safety
- Camera angle, lighting, glasses, and face occlusion can affect detection.
- EAR and head-position rules may produce false positives or false negatives.
- This academic prototype must not be relied on to determine whether someone is safe to drive.
- If sleepy, the driver should stop at a safe location and rest.
References
1. World Health Organization (WHO), Road Traffic Injuries.
2. National Highway Traffic Safety Administration (NHTSA), Drowsy Driving: Avoid Falling Asleep Behind the Wheel.
3. Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection. CVPR Workshops.
4. Google Research / MediaPipe, facial landmark detection documentation.
