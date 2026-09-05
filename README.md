

> **Built for Hacksynapse Hackathon | Team: Code Thrust**  
> An ultra-fast, lightweight, and robust Physical AI edge face recognition engine designed specifically for CPU environments with interactive dynamic multi-angle enrollment.

---

## 🌟 Key Highlights

- ⚡ **CPU-Optimized Real-Time Inference**: Powered by lightweight ONNX runtime execution (`det_500m.onnx` + `w600k_mbf.onnx`) with optimized detection scaling (`320x320`), enabling high-FPS webcam inference without requiring a dedicated GPU.
- 🎯 **Interactive Multi-Angle Live Enrollment**:
  - Manual **`[SPACE]`** / **`[c]`** key trigger to capture distinct facial poses (Front, Profiles, Tilts, Expressions).
  - Built-in **`[y/n]` Verification System** (Accept / Retake) to eliminate duplicate frames and prevent training on blurry or incorrect data.
- 🔄 **Dynamic Zero-Downtime Registration**: Add new identities or expand reference samples for existing persons on-the-fly directly inside the live webcam session without restarting the script.
- 📐 **High-Precision Cosine Metric Matching**: Computes 512-dimensional L2-normalized deep feature vectors and matches identities using vector dot products against a multi-sample gallery.
- 🛡️ **Edge & Physical AI Ready**: Low footprint and modular architecture suitable for deployment on edge devices, smart gates, attendance monitors, and IoT security terminals.

---

## 🏗️ System Architecture & Workflow

```
 ┌─────────────────┐
 │   Camera Feed   │ (Live OpenCV Stream)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Face Detection  │  SCRFD Light (det_500m.onnx @ 320x320)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Feature Extr.   │  ArcFace MobileFaceNet (512-D L2 Normalized)
 └────────┬────────┘
          │
          ├───────────────────────────────┬──────────────────────────────┐
          │                               │                              │
          ▼                               ▼                              ▼
 ┌─────────────────┐             ┌─────────────────┐            ┌─────────────────┐
 │ Real-Time Match │             │ Manual Capture  │            │ Quality Check   │
 │ Cosine Dot Prod │             │ (SPACE / c key) │            │ [y] Save / [n]  │
 └────────┬────────┘             └────────┬────────┘            └────────┬────────┘
          │                               │                              │
          ▼                               └──────────────┬───────────────┘
 ┌─────────────────┐                                     │
 │  Visual Overlay │                                     ▼
 │ Green: Verified │                            ┌─────────────────┐
 │ Red: Unknown    │                            │ Dynamic Gallery │
 └─────────────────┘                            │ Update (Memory) │
                                                └─────────────────┘
```

---

## 📂 Project Structure

```text
Code_Thrust_Hacksynapse_hackathon_physical_ai/
│
├── recognize_cpu.py          # Core Recognition Engine & Live Interactive Controller
├── known_faces/              # Identity Reference Gallery
│   ├── Harish/               # Reference face angles (.jpg, .jpeg, .png)
│   ├── Harshita/             
│   ├── Prateek/              
│   └── Sakshi/               
├── requirements.txt          # Project dependencies
├── .gitignore                # Git exclusions
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HARISH-RAJAK/Code_Thrust_Hacksynapse_hackathon_physical_ai.git
cd Code_Thrust_Hacksynapse_hackathon_physical_ai
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Run

Launch the main controller:
```bash
python recognize_cpu.py
```

You will see an interactive menu:
```text
--- InsightFace CPU Recognizer ---
1. Start Live Webcam Recognition (with Dynamic Enrollment)
2. Test on an Image File
3. Exit
Select option (1/2/3): 
```

### Option 1: Live Webcam / Mobile IP Camera Recognition & Dynamic Enrollment
- **Live HUD Display**: Real-time bounding boxes with detected names, confidence scores, FPS, and stream orientation angle.
- **Controls & Hotkeys**:
  | Key | Action |
  | :--- | :--- |
  | **`r`** | **Rotate Stream Orientation** (`0° -> 90° -> 180° -> 270°`) live on-the-fly |
  | **`e`** / **`n`** | **Trigger Live Enrollment** (Enter person's name in terminal) |
  | **`SPACE`** / **`c`** | **Capture Angle** during enrollment mode |
  | **`y`** | **Accept & Save** captured photo |
  | **`n`** | **Retake / Discard** captured photo |
  | **`q`** / **`ESC`** | **Quit current mode / Exit** |

### Option 2: Test on Static Image File
- Pass any image path (e.g. `test_group.jpg`).
- The script detects all faces, labels matches, and saves an annotated result image to `output_result.jpg`.

---

## 🔬 Algorithm & Matching Details

1. **Face Representation**:
   $$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$
2. **Similarity Scoring**:
   $$\text{Score} = \max_{j} \left( \hat{\mathbf{e}}_{\text{query}} \cdot \hat{\mathbf{e}}_{\text{gallery}, j} \right)$$
3. **Thresholding Rule**:
   - If $\text{Score} \ge 0.45$: Identity is classified as the matched person.
   - If $\text{Score} < 0.45$: Identity is classified as **"Unknown"**.

---

## 👥 Team & Acknowledgments

- **Team Name**: Code Thrust
- **Event**: Hacksynapse Hackathon
- **Developed by**: Harish Rajak & Team

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
