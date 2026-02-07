import os

def get_image_path_from_txt():
    root_env = os.environ.get('PYTHONPATH', '').split(':')[0]

    if not root_env:
        root_env = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal"
    
    txt_file = os.path.join(root_env, "img_path.txt")
    
    if os.path.exists(txt_file):
        with open(txt_file, "r", encoding="utf-8") as f:
            path = f.read().strip()
            return path
    return None