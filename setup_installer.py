import subprocess
import sys

def build_local_desktop_app():
    """
    Automates PyInstaller packaging to compile the multi-modal child safety 
    triage pipeline into a single, localized executable desktop installer.
    """
    print("🚀 Initiating Local Desktop Application Compilation Layer...")
    
    # Install PyInstaller inside the local machine sandbox
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # PyInstaller execution command:
    # --onefile: Bundles all code and libraries into a single executable file.
    # --hidden-import: Ensures heavy neural network libraries (PyTorch/Transformers) are fully mapped.
    build_command = [
        "pyinstaller",
        "--onefile",
        "--name=ChildSafetyTriageGuard",
        "--hidden-import=torch",
        "--hidden-import=transformers",
        "--hidden-import=streamlit",
        "app.py"
    ]
    
    try:
        subprocess.run(build_command, check=True)
        print("✅ Success! Local installer created inside the 'dist/' folder.")
        print("👉 Parents can now double-click 'ChildSafetyTriageGuard' to run the privacy-first pipeline locally.")
    except Exception as e:
        print(f"❌ Desktop Packaging Interrupted: {str(e)}")

if __name__ == "__main__":
    build_local_desktop_app()
