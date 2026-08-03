# 🛡️ Child Safety Multimodal Triage Dashboard
### Developed by Srinivasta

An advanced, privacy-first Trust & Safety engineering solution built to automate the triage, classification, and analysis of multi-modal data streams (Text and Vision). This architecture integrates code-safe synthetic scenario simulations with dynamic, zero-shot open-vocabulary metric projections to intercept child safety violations without relying on static keyword lists.

---

## 🏗️ Repository Architecture Tree

```text
child-safety-classifier/
│
├── .github/workflows/
│   └── build-exe.yml     # GitHub Actions cloud compilation engine (Windows EXE builder)
├── config.py             # Centralized model specifications and embedding hyper-parameters
├── model.py              # Multimodal metric-space projection bridge (CLIP + DistilBERT)
├── utils.py              # Cryptographic SHA-256 PII anonymization & logging layer
├── app.py                # Streamlit Master Web UI + In-Browser REST API JSON Simulator
├── api.py                # Standalone FastAPI REST web server layer for automated platform hooks
├── setup_installer.py    # Local compiler script mapping Python modules to desktop binaries
└── requirements.txt      # Pinned core machine learning deployment dependencies
```

---

## ⚡ Core Engineering Profiles

### 🌐 1. Cloud Web Ingestion (`app.py`)
Runs natively as an interactive web dashboard on **Streamlit Community Cloud**. It provides human operators with dual-modality triage columns, real-time alert sensitivity cutoff sliders, a "Demo Safe" verification override, and an instantaneous visual confirmation monitor channel.

### 🤖 2. Open-Vocabulary Vector Automation (`api.py`)
Utilizes a **Late-Fusion Metric Space Bridge** (`model.py`) to map raw inputs (`DistilBERT` text embeddings and `CLIP` visual tensors) side-by-side into a shared coordinate grid. By measuring real-time mathematical cosine similarities against stable reference anchors, it handles unhandled strings (like *"dog in a house"*) as clear, avoiding common random-weight false alarms.

### 🔌 3. Machine-to-Machine Integration Console
Embedded natively within the web UI dashboard, this module exposes automated **REST API JSON payloads**. It enables seamless background translation into external protection tools (such as automated YouTube description scanners, Discord webhooks, or forum registration validation hooks).

### 🔒 4. Edge-Computing Desktop Privacy Sandbox (`setup_installer.py`)
Compiles the complete deep learning project framework into a portable Windows application file (`ChildSafetyTriageGuard.exe`). Because it runs entirely on-device, **no information ever leaves the machine**, providing maximum data privacy for family computing environments.

---

## 💻 Local Installation & Setup

If you wish to download and run Srinivasta's application natively on your physical computer, execute the following commands inside your terminal:

```bash
# Clone or extract the repository source core
cd child-safety-classifier

# Configure a sandboxed Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows PowerShell use: .\venv\Scripts\activate

# Install machine learning dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch the human triage graphical dashboard interface
streamlit run app.py
```

---

## 🚀 Cloud Executable Compilation Workflow (`build-exe.yml`)

This project implements an automated **GitHub Actions CI/CD pipeline** to bypass cloud OS boundaries. Every time you push a code modification on GitHub Web, a remote Microsoft Windows Server container spins up in the cloud, invokes PyInstaller via `setup_installer.py`, and bundles the app, Python engine, and weights together. 

### 📥 How to Download the Finished App:
1. Navigate to the **Actions** tab at the top of your GitHub repository web page.
2. Select the latest green-stamped workflow build run.
3. Scroll down to the **Artifacts** section and click **`ChildSafetyGuard-Windows-EXE`**.
4. Parents can double-click this `.exe` file to boot the child safety guard system with **no Python installation or coding background required**.

---
*Maintained under secure open-source guidelines supporting modern Trust & Safety infrastructure engineering.*
