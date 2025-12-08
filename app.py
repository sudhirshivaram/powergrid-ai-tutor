"""
Symlink or wrapper script for HuggingFace Spaces deployment.
This ensures the app runs from the correct entry point.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and launch the main app
from app.main_with_api_key import PowerGridTutorUI

if __name__ == "__main__":
    tutor = PowerGridTutorUI()
    tutor.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
