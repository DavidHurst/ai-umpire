__all__ = ["VideoGenerator"]

from pathlib import Path
import yaml


class VideoGenerator:
    def __init__(self, path_to_data_dir: str = '../data'):
        self.data_dir_path: Path = Path(path_to_data_dir)
        try:
            self.data_dir_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")

        # Check config file exist
        if not (Path('.') / 'config.yaml').exists():
            raise FileNotFoundError("Config file not found")
        else:
            self.settings_fpath: Path = Path('.') / 'config.yaml'

    def add_new_sample_settings(self) -> None:
        # Append new configs to respective files
        pass

    def generate_video(self, sample_id) -> None:
        # Load settings for given sample ID and generate video
        with open(self.settings_fpath, "r") as f:
            settings_dict: dict = yaml.safe_load(f)
        print(settings_dict.get(sample_id))

    def generate_all_videos(self) -> None:
        # Call generate_video for all samples
        pass
