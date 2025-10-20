import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root_dir = os.path.join(_current_dir.split("risky_navigation")[0] , "risky_navigation")

SRC_PATH = os.path.join(_proj_root_dir, r"src")
LEARNING_PATH = os.path.join(SRC_PATH, r"learning")
