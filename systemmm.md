# 🏗️ System Architecture

## AI-Based Event Monitoring & Resource Management System

---

## 1. Architecture Overview

The system follows a **modular pipeline architecture** where each processing stage is isolated into its own module. Data flows sequentially through five stages — from video input to final visualization — with each stage consuming the output of the previous one.

The system supports two execution modes:
- **Dashboard Mode** (Streamlit) — Interactive web UI with real-time streaming
- **CLI Mode** (Command Line) — Batch processing with terminal output

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                       │
│                                                                 │
│   ┌─────────────────────┐      ┌──────────────────────┐        │
│   │   dashboard.py      │      │      main.py         │        │
│   │   (Streamlit UI)    │      │      (CLI)           │        │
│   │                     │      │                      │        │
│   │  • File Upload      │      │  • argparse flags    │        │
│   │  • URL Paste        │      │  • Console output    │        │
│   │  • Live CCTV        │      │  • CSV export        │        │
│   │  • Webcam           │      │                      │        │
│   │  • YouTube Live     │      │                      │        │
│   └────────┬────────────┘      └──────────┬───────────┘        │
│            │                              │                     │
└────────────┼──────────────────────────────┼─────────────────────┘
             │                              │
             ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                         │
│                         (modules/)                              │
│                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│   │ downloader.py│  │  tracker.py  │  │    detector.py       │ │
│   │              │  │              │  │                      │ │
│   │ Video URL    │──│ Frame-by-    │──│ YOLOv8 Nano          │ │
│   │ Download &   │  │ frame read   │  │ People Detection     │ │
│   │ Stream       │  │ & process    │  │                      │ │
│   │ Extraction   │  │              │  │ Input: BGR frame     │ │
│   └──────────────┘  └──────┬───────┘  │ Output: boxes, count │ │
│                            │          └──────────────────────┘ │
│                            ▼                                    │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │                   ANALYSIS LAYER                         │ │
│   │                                                          │ │
│   │   ┌──────────────┐         ┌───────────────────────┐    │ │
│   │   │   risk.py    │────────►│   resources.py        │    │ │
│   │   │              │         │                       │    │ │
│   │   │ • Density    │         │ • Guard deployment    │    │ │
│   │   │ • Count Δ    │         │ • Water estimation    │    │ │
│   │   │ • Risk score │         │ • Food estimation     │    │ │
│   │   │ • Risk level │         │                       │    │ │
│   │   └──────────────┘         └───────────────────────┘    │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│                                                                 │
│   ┌──────────────────┐    ┌─────────────────────────────────┐  │
│   │  Pandas          │    │  YOLOv8 Model Weights           │  │
│   │  DataFrame       │    │  (yolov8n.pt — 6.5 MB)          │  │
│   │  (in-memory)     │    │  Pre-trained on COCO dataset    │  │
│   └──────────────────┘    └─────────────────────────────────┘  │
│                                                                 │
│   ┌──────────────────┐    ┌─────────────────────────────────┐  │
│   │  CSV Export      │    │  Temp Files                     │  │
│   │  (output/)       │    │  (downloaded videos, uploads)   │  │
│   └──────────────────┘    └─────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Flow

### 2.1 Component Interaction Diagram

```
                    ┌──────────────────────────────┐
                    │         USER / CLIENT         │
                    │   (Browser or Terminal)       │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     ENTRY POINT SELECTION     │
                    │                               │
                    │  dashboard.py    main.py      │
                    │  (Web UI)       (CLI)         │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────────┐
     │ File Upload│   │  URL Input │   │  Live Stream   │
     │            │   │            │   │  (CCTV/Webcam/ │
     │ tempfile   │   │ downloader │   │  YouTube Live) │
     │ storage    │   │ .py        │   │                │
     └─────┬──────┘   └─────┬──────┘   └──────┬─────────┘
           │                │                  │
           └────────────────┼──────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   tracker.py    │
                   │                 │
                   │ • Opens video   │
                   │   via OpenCV    │
                   │ • Reads frames  │
                   │ • Applies frame │
                   │   skipping      │
                   │ • Calls         │
                   │   detector      │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  detector.py    │
                   │                 │
                   │ • YOLOv8 Nano   │
                   │   inference     │
                   │ • Filter for    │
                   │   class 0       │
                   │   (person)      │
                   │ • Draw boxes    │
                   │ • Return count  │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │                 │
                   │  Per-frame      │
                   │  record:        │
                   │  {frame,        │
                   │   timestamp,    │
                   │   people_count} │
                   │                 │
                   └────────┬────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
      ┌──────────────┐         ┌──────────────────┐
      │ STREAMING    │         │ BATCH MODE       │
      │ MODE         │         │ (CLI)            │
      │ (Dashboard)  │         │                  │
      │              │         │ Collect all      │
      │ Yield each   │         │ records into     │
      │ frame to UI  │         │ DataFrame first  │
      │ in real-time │         │                  │
      └──────┬───────┘         └────────┬─────────┘
             │                          │
             └────────────┬─────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    risk.py      │
                 │                 │
                 │ • crowd_density │
                 │ • count_change  │
                 │ • risk_score    │
                 │ • risk_level    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  resources.py   │
                 │                 │
                 │ • guards_needed │
                 │ • water_demand  │
                 │ • food_demand   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   OUTPUT        │
                 │                 │
                 │ • Dashboard:    │
                 │   Charts,       │
                 │   Metrics,      │
                 │   Data Table    │
                 │                 │
                 │ • CLI:          │
                 │   Console       │
                 │   summary,      │
                 │   CSV file      │
                 └─────────────────┘
```

### 2.2 Module Dependency Graph

Shows which module imports from which:

```
dashboard.py ────────┬──── modules/tracker.py ──── modules/detector.py
                     │                                    │
                     ├──── modules/risk.py                │
                     │                                    │
                     ├──── modules/resources.py           │
                     │                                    │
                     └──── modules/downloader.py          │
                                                          │
main.py ─────────────┬──── modules/tracker.py ────────────┘
                     │
                     ├──── modules/risk.py
                     │
                     ├──── modules/resources.py
                     │
                     └──── modules/downloader.py


External Dependencies:
  detector.py    →  ultralytics (YOLO), cv2 (OpenCV)
  tracker.py     →  cv2 (OpenCV), pandas
  risk.py        →  numpy
  resources.py   →  math (built-in)
  downloader.py  →  subprocess, urllib, tempfile, shutil (all built-in)
  dashboard.py   →  streamlit, matplotlib, cv2, pandas, tempfile
  main.py        →  argparse, pandas (all built-in)
```

### 2.3 Processing Modes

The system operates in three distinct processing modes depending on the input type:

| Mode | Entry Point | Function Used | Behavior |
|------|-------------|---------------|----------|
| **Batch (CLI)** | `main.py` | `process_video()` | Reads entire video, returns full DataFrame |
| **Streaming (Dashboard)** | `dashboard.py` | `process_video_streaming()` | Generator — yields `(frame, record, progress)` per frame |
| **Live (Dashboard)** | `dashboard.py` | `process_live_stream()` | Generator — yields `(frame, record)` indefinitely until stopped |

---

## 3. Data Flow

### 3.1 End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: VIDEO INPUT                                                    │
│                                                                         │
│  Input Sources:                                                         │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────────────┐    │
│  │ MP4/AVI │ │YouTube  │ │  RTSP/   │ │ Webcam │ │  YouTube      │    │
│  │ File    │ │URL      │ │  HTTP    │ │ Device │ │  Livestream   │    │
│  │ Upload  │ │Download │ │  Stream  │ │ Index  │ │  URL          │    │
│  └────┬────┘ └────┬────┘ └────┬─────┘ └───┬────┘ └──────┬────────┘    │
│       │           │           │            │             │              │
│       │      ┌────▼────┐      │            │        ┌────▼──────┐      │
│       │      │yt-dlp / │      │            │        │ yt-dlp    │      │
│       │      │urllib   │      │            │        │ --get-url │      │
│       │      │download │      │            │        │ extract   │      │
│       │      └────┬────┘      │            │        └────┬──────┘      │
│       │           │           │            │             │              │
│       ▼           ▼           ▼            ▼             ▼              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              cv2.VideoCapture(source)                       │       │
│  │         Opens video file / stream / device                  │       │
│  └─────────────────────────┬───────────────────────────────────┘       │
└────────────────────────────┼────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: FRAME EXTRACTION                                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    Frame Loop                               │       │
│  │                                                             │       │
│  │   cap.read() → ret, frame (BGR NumPy array)                │       │
│  │                                                             │       │
│  │   Frame Skip Logic:                                         │       │
│  │   if frame_number % frame_skip != 0:                        │       │
│  │       skip → next frame                                     │       │
│  │   else:                                                     │       │
│  │       process → detect_people(frame)                        │       │
│  │                                                             │       │
│  └─────────────────────────┬───────────────────────────────────┘       │
│                            │                                            │
│   Data at this stage:      │                                            │
│   • Raw BGR frame          │                                            │
│     (NumPy array           │                                            │
│      H × W × 3)           │                                            │
│                            │                                            │
└────────────────────────────┼────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DETECTION (detector.py)                                        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  model = YOLO("yolov8n.pt")                             │          │
│  │  results = model(frame, verbose=False)                   │          │
│  │                                                          │          │
│  │  For each detection:                                     │          │
│  │    class_id = int(box.cls[0])                            │          │
│  │    if class_id == 0:   ← (person class)                  │          │
│  │        Extract (x1, y1, x2, y2) bounding box             │          │
│  │        Extract confidence score                           │          │
│  │        Draw green rectangle on frame                      │          │
│  │        Draw "Person 0.85" label above box                 │          │
│  │                                                          │          │
│  │  Draw "People: N" on top-left corner                     │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
│   Output at this stage:                                                 │
│   • boxes:           [(x1,y1,x2,y2), ...]   ← bounding box coords     │
│   • annotated_frame: BGR array with drawings ← for display             │
│   • count:           int                     ← number of people        │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: RECORD CREATION (tracker.py)                                   │
│                                                                         │
│   record = {                                                            │
│       "frame":         frame_number,          ← e.g., 42               │
│       "timestamp_sec": frame_number / fps,    ← e.g., 1.40             │
│       "people_count":  count                  ← e.g., 12               │
│   }                                                                     │
│                                                                         │
│   ┌──────────────────────────────────────────────┐                     │
│   │  In streaming mode (dashboard):              │                     │
│   │  yield (annotated_frame, record, progress)   │                     │
│   │         ↓                ↓          ↓        │                     │
│   │     Displayed        Appended   Progress     │                     │
│   │     in UI            to list    bar update   │                     │
│   └──────────────────────────────────────────────┘                     │
│                                                                         │
│   ┌──────────────────────────────────────────────┐                     │
│   │  In batch mode (CLI):                        │                     │
│   │  records.append(record)                      │                     │
│   │  → All records collected into DataFrame      │                     │
│   └──────────────────────────────────────────────┘                     │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: RISK CALCULATION (risk.py)                                     │
│                                                                         │
│   Input DataFrame:                                                      │
│   ┌───────┬───────────────┬──────────────┐                             │
│   │ frame │ timestamp_sec │ people_count │                             │
│   ├───────┼───────────────┼──────────────┤                             │
│   │ 0     │ 0.00          │ 5            │                             │
│   │ 3     │ 0.10          │ 8            │                             │
│   │ 6     │ 0.20          │ 12           │                             │
│   └───────┴───────────────┴──────────────┘                             │
│                                                                         │
│   Transformations:                                                      │
│                                                                         │
│   1. crowd_density = (people_count / 2073600) × 10000                  │
│      ↑ frame_area = 1920 × 1080 = 2,073,600 pixels                    │
│                                                                         │
│   2. count_change = |people_count - previous_people_count|             │
│                                                                         │
│   3. Normalize both to 0–1 range:                                      │
│      norm_density = crowd_density / max(crowd_density)                 │
│      norm_change  = count_change / max(count_change)                   │
│                                                                         │
│   4. risk_score = (0.6 × norm_density) + (0.4 × norm_change)          │
│                    ↑ density weight       ↑ surge weight               │
│                                                                         │
│   5. risk_level = threshold(risk_score):                               │
│      ┌────────────────┬────────────────┐                               │
│      │ Score Range    │ Level          │                               │
│      ├────────────────┼────────────────┤                               │
│      │ 0.00 – 0.24   │ Low            │                               │
│      │ 0.25 – 0.49   │ Medium         │                               │
│      │ 0.50 – 0.74   │ High           │                               │
│      │ 0.75 – 1.00   │ Critical       │                               │
│      └────────────────┴────────────────┘                               │
│                                                                         │
│   Output DataFrame (4 new columns added):                              │
│   ┌───────┬──────────┬──────────────┬────────────────┬──────────────┐  │
│   │ frame │ ... (3)  │crowd_density │ count_change   │ risk_score   │  │
│   ├───────┤          ├──────────────┤ ──────────────┤──────────────┤  │
│   │ ...   │          │ 0.0241       │ 0              │ 0.350        │  │
│   └───────┘          └──────────────┘ ──────────────┘──────────────┘  │
│   + risk_level column (Low/Medium/High/Critical)                       │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: RESOURCE ESTIMATION (resources.py)                             │
│                                                                         │
│   6a. Guard Deployment:                                                 │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  For each row:                                           │         │
│   │    if people_count == 0 → guards_needed = 0              │         │
│   │    else:                                                 │         │
│   │      ratio = GUARD_RATIOS[risk_level]                    │         │
│   │      guards_needed = max(1, ceil(people_count / ratio))  │         │
│   │                                                          │         │
│   │  GUARD_RATIOS:                                           │         │
│   │    Low      → 50  (1 guard per 50 people)                │         │
│   │    Medium   → 20  (1 guard per 20 people)                │         │
│   │    High     → 10  (1 guard per 10 people)                │         │
│   │    Critical →  7  (1 guard per 7 people)                 │         │
│   └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│   6b. Resource Demand:                                                  │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  water_demand = people_count × 0.5 liters                │         │
│   │  food_demand  = people_count × 0.6 meals                 │         │
│   └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│   Final DataFrame (3 new columns added):                               │
│   All 10 columns: frame, timestamp_sec, people_count,                  │
│   crowd_density, count_change, risk_score, risk_level,                 │
│   guards_needed, water_demand, food_demand                             │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: OUTPUT & VISUALIZATION                                         │
│                                                                         │
│   Dashboard (Streamlit):                                                │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  • Key Metrics        → 4 metric cards                   │         │
│   │  • People Count Chart → Line + area fill                 │         │
│   │  • Risk Score Chart   → Color-coded scatter plot         │         │
│   │  • Guard Deployment   → Bar chart + ratio caption        │         │
│   │  • Resource Table     → Per-person × crowd = total       │         │
│   │  • Data Table         → Full DataFrame + CSV download    │         │
│   └──────────────────────────────────────────────────────────┘         │
│                                                                         │
│   CLI:                                                                  │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │  • Console summary    → Key stats printed to terminal    │         │
│   │  • CSV export         → output/results.csv               │         │
│   │  • Annotated video    → output/output_video.mp4          │         │
│   │                         (optional, --save-video flag)     │         │
│   └──────────────────────────────────────────────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 DataFrame Column Evolution

This table shows how the DataFrame grows as it passes through each module:

| Stage | Module | Columns Added | Total Columns |
|-------|--------|---------------|---------------|
| 1. Frame Processing | `tracker.py` | `frame`, `timestamp_sec`, `people_count` | 3 |
| 2. Risk Calculation | `risk.py` | `crowd_density`, `count_change`, `risk_score`, `risk_level` | 7 |
| 3. Guard Deployment | `resources.py` | `guards_needed` | 8 |
| 4. Resource Estimation | `resources.py` | `water_demand`, `food_demand` | **10** |

### 3.3 Data Types & Ranges

| Column | Type | Range / Values | Example |
|--------|------|----------------|---------|
| `frame` | `int` | 0 to total_frames | `42` |
| `timestamp_sec` | `float` | 0.0 to video_duration | `1.40` |
| `people_count` | `int` | 0 to ∞ | `12` |
| `crowd_density` | `float` | 0.0 to ~1.0 | `0.0579` |
| `count_change` | `int` | 0 to ∞ | `3` |
| `risk_score` | `float` | 0.0 to 1.0 | `0.650` |
| `risk_level` | `str` | Low / Medium / High / Critical | `"High"` |
| `guards_needed` | `int` | 0 to ∞ | `2` |
| `water_demand` | `float` | 0.0 to ∞ | `6.0` |
| `food_demand` | `float` | 0.0 to ∞ | `7.2` |

---

## 4. Live Stream Architecture

Live streams use a different architecture pattern than file-based processing. Key differences:

```
                FILE-BASED PROCESSING                    LIVE STREAM PROCESSING
                ─────────────────────                    ──────────────────────

             ┌──────────────────┐                     ┌──────────────────┐
             │ Read all frames  │                     │ Read frames      │
             │ sequentially     │                     │ indefinitely     │
             └────────┬─────────┘                     └────────┬─────────┘
                      │                                        │
                      ▼                                        ▼
             ┌──────────────────┐                     ┌──────────────────┐
             │ Collect ALL      │                     │ Yield each       │
             │ records into     │                     │ frame + record   │
             │ records list     │                     │ immediately      │
             └────────┬─────────┘                     └────────┬─────────┘
                      │                                        │
                      ▼                                        ▼
             ┌──────────────────┐                     ┌──────────────────┐
             │ Build DataFrame  │                     │ Append to rolling│
             │ once at end      │                     │ buffer (max 500) │
             └────────┬─────────┘                     └────────┬─────────┘
                      │                                        │
                      ▼                                        ▼
             ┌──────────────────┐                     ┌──────────────────┐
             │ Run risk +       │                     │ Every 30 frames: │
             │ resources once   │                     │ Run risk +       │
             └────────┬─────────┘                     │ resources on     │
                      │                               │ rolling buffer   │
                      ▼                               └────────┬─────────┘
             ┌──────────────────┐                              │
             │ Render final     │                              ▼
             │ charts           │                     ┌──────────────────┐
             └──────────────────┘                     │ Update charts    │
                                                      │ in-place         │
                                                      └────────┬─────────┘
                                                               │
                                                               ▼
                                                      ┌──────────────────┐
                                                      │ On stop:         │
                                                      │ Final analysis   │
                                                      │ + full render    │
                                                      └──────────────────┘
```

### Live Stream Resilience

The `process_live_stream()` function includes automatic reconnection:

```
                    ┌──────────────┐
                    │ cap.read()   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ ret == True? │
                    └──────┬───────┘
                      Yes  │   No
                    ┌──────┘   └──────┐
                    ▼                 ▼
             ┌────────────┐   ┌─────────────┐
             │ Process    │   │ cap.release()│
             │ frame      │   │ Reconnect   │
             └────────────┘   │ to source   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │ Connected?  │
                              └──────┬──────┘
                                Yes  │  No
                              ┌──────┘  └──────┐
                              ▼                ▼
                       ┌────────────┐   ┌────────────┐
                       │ Continue   │   │ Break loop │
                       │ reading    │   │ (end)      │
                       └────────────┘   └────────────┘
```

---

## 5. Streamlit Dashboard Layout Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ┌─────────────────┐  ┌─────────────────────────────────────────────┐   │
│ │   SIDEBAR       │  │                MAIN AREA                    │   │
│ │                 │  │                                             │   │
│ │ ┌─────────────┐ │  │  ┌───────────────────────────────────────┐ │   │
│ │ │📂 Video     │ │  │  │  📹 Title & Description              │ │   │
│ │ │   Input     │ │  │  └───────────────────────────────────────┘ │   │
│ │ │             │ │  │                                             │   │
│ │ │ ○ Upload    │ │  │  ┌───────────────────────────────────────┐ │   │
│ │ │ ○ Paste URL │ │  │  │  🔍 Live Frame Display               │ │   │
│ │ │ ○ CCTV      │ │  │  │  (real-time annotated video)         │ │   │
│ │ │ ○ Webcam    │ │  │  └───────────────────────────────────────┘ │   │
│ │ │ ○ YT Live   │ │  │                                             │   │
│ │ └─────────────┘ │  │  ┌───────────────────────────────────────┐ │   │
│ │                 │  │  │  📊 Key Metrics (4 columns)           │ │   │
│ │ ┌─────────────┐ │  │  │  Max People | Avg | Risk | Level     │ │   │
│ │ │⚙️ Settings  │ │  │  └───────────────────────────────────────┘ │   │
│ │ │             │ │  │                                             │   │
│ │ │ Frame Skip  │ │  │  ┌────────────────┐  ┌────────────────┐   │   │
│ │ │ [slider]    │ │  │  │ People Count   │  │ Risk Score     │   │   │
│ │ │             │ │  │  │ Over Time      │  │ Over Time      │   │   │
│ │ └─────────────┘ │  │  │ (line chart)   │  │ (scatter plot) │   │   │
│ │                 │  │  └────────────────┘  └────────────────┘   │   │
│ │ ┌─────────────┐ │  │                                             │   │
│ │ │ Start/Stop  │ │  │  ┌───────────────────────────────────────┐ │   │
│ │ │ Buttons     │ │  │  │  🛡️ Guards Needed Over Time          │ │   │
│ │ │ (live only) │ │  │  │  (bar chart, full width)              │ │   │
│ │ └─────────────┘ │  │  └───────────────────────────────────────┘ │   │
│ │                 │  │                                             │   │
│ │                 │  │  ┌───────────────────────────────────────┐ │   │
│ │                 │  │  │  🍽️ Resource Consumption Estimates   │ │   │
│ │                 │  │  │  3 metrics + reasoning table          │ │   │
│ │                 │  │  └───────────────────────────────────────┘ │   │
│ │                 │  │                                             │   │
│ │                 │  │  ┌───────────────────────────────────────┐ │   │
│ │                 │  │  │  📋 Detailed Results                 │ │   │
│ │                 │  │  │  DataFrame + CSV download             │ │   │
│ │                 │  │  └───────────────────────────────────────┘ │   │
│ └─────────────────┘  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Memory & Performance Characteristics

| Aspect | File Mode | Live Stream Mode |
|--------|-----------|-----------------|
| **Frame Buffer** | All processed frames stored | Rolling window of 500 frames max |
| **Chart Updates** | Once, after completion | Every 30 processed frames |
| **Model Loading** | Once at module import | Once at module import |
| **Video I/O** | Sequential read, finite | Continuous read, indefinite |
| **Temp Files** | Uploaded files → temp dir → deleted after processing | No temp files (direct stream) |
| **Memory Growth** | Linear with video length | Bounded by rolling window |
