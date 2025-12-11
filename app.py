"""
HuggingFace Spaces entry point for PowerGrid AI Tutor.
"""

import sys
import warnings
from pathlib import Path

# Suppress noisy asyncio warnings
warnings.filterwarnings("ignore", message="Invalid file descriptor")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and launch the main app
from app.main import PowerGridTutorUI

if __name__ == "__main__":
 tutor = PowerGridTutorUI()
 interface = tutor.create_interface()
 # Launch with queue enabled for HuggingFace Spaces
 interface.queue()
 interface.launch()
