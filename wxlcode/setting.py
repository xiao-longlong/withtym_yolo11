import os
import sys
os.getcwd()
sys.path.insert(0, os.getcwd())

from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]