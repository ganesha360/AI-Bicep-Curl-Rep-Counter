# 🏋️‍♂️ AI Bicep Curl Rep Counter

A real-time AI fitness assistant that uses computer vision to track bicep curls, count reps, and measure tempo — just like a personal trainer.

---

## ✨ Features

> 📦 **Feature Overview**

| Feature | Description |
|--------|-------------|
| 💪 **Dual Arm Tracking** | Detects and isolates left and right arms. |
| 🔢 **Smart Rep Counting** | Uses elbow angle thresholds (Up < 80°, Down > 160°) for accurate reps. |
| ⏱️ **Tempo Analysis** | Measures concentric time-under-tension during each lift. |
| 🌊 **Signal Smoothing** | Moving average filtering to reduce jitter and false triggers. |
| 📊 **Live Dashboard** | Displays angles, reps, and tracking status in real-time. |
| 📈 **Session Summary** | Generates a workout report after exit. |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **OpenCV**
- **MediaPipe Pose**
- **NumPy**

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ganesha360/Ai-Bicep-Curl-Rep-Counter.git
cd Ai-Bicep-Curl-Rep-Counter
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

### 3️⃣ Activate the Environment

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Usage

### 🚀 Run the Tracker

```bash
python fitness_coach.py
```

### Positioning

- Stand **2–3 meters** from webcam.
- Ensure upper body is fully visible.
- HUD text will show **Tracking Active** when ready.

### Controls

| Action | Key |
|--------|-----|
| Quit & export summary | `q` |

---

## 🔍 How It Works

1. **Pose Detection:** MediaPipe extracts 33 skeleton keypoints.
2. **Angle Calculation:** Shoulder–elbow–wrist angle is computed using vector geometry.
3. **State Machine Logic:**
   - **Down position:** angle > 160°
   - **Up position:** angle < 80°
   - Rep counted only after **Down → Up → Down** cycle.
4. **Tempo Measurement:** Concentric duration recorded and averaged.

---

## 👤 Author

**👨‍💻 GANESH R**

📩 **Email:** ganeshravi360@gmail.com  
🔗 **LinkedIn:** [linkedin.com/in/ganesharavi](https://linkedin.com/in/ganesharavi)  
🌐 **Portfolio:** [ganesha360.github.io/portfolio](https://ganesha360.github.io/portfolio/)



---

## 🤝 Contributing

Contributions are welcome and appreciated.  
Please fork the repository and submit a pull request.

Before contributing, review the project structure and follow the existing code style.

---
