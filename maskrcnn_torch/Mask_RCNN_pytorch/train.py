import json
from pathlib import Path
from config import json_file_path
from training import train_model 

with open(json_file_path) as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]


def main():
    train_model()

if __name__ == "__main__":
    main()
