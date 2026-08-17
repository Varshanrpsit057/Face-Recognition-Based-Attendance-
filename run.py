import os
import sys
import subprocess
import pathlib
import platform
import venv
import urllib.request
import time
import webbrowser

# Enable Windows ANSI escape sequences if applicable
if platform.system() == "Windows":
    os.system("")

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    banner = f"""{Colors.OKCYAN}{Colors.BOLD}
======================================================
     FACIAL RECOGNITION ATTENDANCE SYSTEM SETUP
======================================================
{Colors.ENDC}"""
    print(banner)

def check_python():
    print(f"{Colors.OKBLUE}[*] Checking Python version...{Colors.ENDC}")
    if sys.version_info < (3, 9):
        print(f"{Colors.FAIL}[!] Python 3.9 or higher is required. Found {sys.version_info.major}.{sys.version_info.minor}.{Colors.ENDC}")
        sys.exit(1)
    print(f"{Colors.OKGREEN}[+] Python version is compatible.{Colors.ENDC}")

def check_internet():
    print(f"{Colors.OKBLUE}[*] Checking Internet connection...{Colors.ENDC}")
    try:
        urllib.request.urlopen("https://1.1.1.1", timeout=3)
        print(f"{Colors.OKGREEN}[+] Internet connection is active.{Colors.ENDC}")
    except Exception:
        print(f"{Colors.WARNING}[!] No internet connection detected. Network-dependent steps may fail.{Colors.ENDC}")

def check_gpu():
    print(f"{Colors.OKBLUE}[*] Checking for GPU (nvidia-smi)...{Colors.ENDC}")
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}[+] NVIDIA GPU detected.{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}[!] nvidia-smi failed, continuing on CPU.{Colors.ENDC}")
    except FileNotFoundError:
        print(f"{Colors.WARNING}[!] nvidia-smi not found, continuing on CPU.{Colors.ENDC}")

def create_venv():
    venv_dir = pathlib.Path('.venv')
    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    # Validate existing venv (detect if copied from another path/computer)
    if venv_dir.exists():
        pyvenv_cfg = venv_dir / "pyvenv.cfg"
        is_invalid = not pyvenv_cfg.exists()
        if not is_invalid:
            try:
                res = subprocess.run([str(python_exe), "-c", "import sys; print(sys.prefix)"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode != 0:
                    is_invalid = True
            except Exception:
                is_invalid = True

        if is_invalid:
            print(f"{Colors.WARNING}[!] Existing .venv is invalid or was copied from another location. Recreating...{Colors.ENDC}")
            import shutil
            shutil.rmtree(venv_dir, ignore_errors=True)

    if not venv_dir.exists():
        print(f"{Colors.OKBLUE}[*] Creating virtual environment...{Colors.ENDC}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)
        print(f"{Colors.OKGREEN}[+] Virtual environment created.{Colors.ENDC}")
    else:
        print(f"{Colors.OKGREEN}[+] Virtual environment verified.{Colors.ENDC}")

    return python_exe, pip_exe

def install_requirements(pip_exe):
    req_file = pathlib.Path("requirements.txt")
    if not req_file.exists():
        print(f"{Colors.WARNING}[!] requirements.txt not found. Skipping installation.{Colors.ENDC}")
        return
        
    print(f"{Colors.OKBLUE}[*] Installing dependencies...{Colors.ENDC}")
    for attempt in range(3):
        print(f"Attempt {attempt + 1}/3...")
        result = subprocess.run([str(pip_exe), "install", "-r", str(req_file)])
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}[+] Dependencies installed successfully.{Colors.ENDC}")
            return
        time.sleep(2)
    print(f"{Colors.FAIL}[!] Failed to install dependencies after 3 attempts.{Colors.ENDC}")

def create_directories():
    print(f"{Colors.OKBLUE}[*] Creating project directories...{Colors.ENDC}")
    dirs = [
        "dataset",
        "models",
        "outputs",
        "outputs/embeddings",
        "outputs/logs",
        "attendance_logs",
        "reports",
        "cache",
        "sample_video"
    ]
    for d in dirs:
        path = pathlib.Path(d)
        path.mkdir(parents=True, exist_ok=True)
    print(f"{Colors.OKGREEN}[+] Directories verified/created.{Colors.ENDC}")

def download_models(python_exe):
    print(f"{Colors.OKBLUE}[*] Downloading models...{Colors.ENDC}")
    code = (
        "import sys; sys.path.insert(0, '.'); from pathlib import Path; "
        "from src.downloader import ModelDownloader; "
        "d = ModelDownloader(Path('models')); d.download_all_models()"
    )
    result = subprocess.run([str(python_exe), "-c", code])
    if result.returncode == 0:
        print(f"{Colors.OKGREEN}[+] Models downloaded successfully.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}[!] Model download failed.{Colors.ENDC}")

def main():
    print_banner()
    check_python()
    check_internet()
    check_gpu()
    
    python_exe, pip_exe = create_venv()
    
    install_requirements(pip_exe)
    create_directories()
    download_models(python_exe)
    
    print(f"{Colors.OKBLUE}[*] Launching application...{Colors.ENDC}")
    try:
        import threading
        def open_browser():
            time.sleep(3)
            webbrowser.open('http://localhost:8501')
        threading.Thread(target=open_browser, daemon=True).start()
        
        subprocess.run([str(python_exe), "-m", "streamlit", "run", "app.py", "--server.headless", "true"])
    except KeyboardInterrupt:
        print(f"\n{Colors.OKGREEN}[+] Application stopped gracefully.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}[!] An error occurred: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()
