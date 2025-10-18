import subprocess
import sys

# Force-install scikit-learn at runtime
try:
    import sklearn
    print("✅ scikit-learn already installed.")
except ImportError:
    print("⚙️ Installing scikit-learn manually...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.4.2"])
    import sklearn
    print("✅ scikit-learn installed successfully:", sklearn.__version__)
