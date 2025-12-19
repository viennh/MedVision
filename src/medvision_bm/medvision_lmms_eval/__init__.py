import os
import sys

# Get the path to the current directory (src/medvision_bm/medvision_lmms_eval)
_vendor_path = os.path.dirname(os.path.abspath(__file__))

# Add it to sys.path so 'import lmms_eval' works internally
if _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)
