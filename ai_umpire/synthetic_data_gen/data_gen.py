__all__ = ["VideoGenerator"]

from pathlib import Path
import pybullet as pb
import yaml
from tqdm import tqdm

ROT_90_X_QUAT = [
    0.7071081,
    0,
    0,
    0.7071055,
]  # Correction for blender using different axes


class VideoGenerator:
    def __init__(self, path_to_data_dir: str):
        self.data_dir_path: Path = Path(path_to_data_dir)
        try:
            self.data_dir_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Data folder already exists.")
        else:
            print("Data folder was created.")

        fpath: Path = Path(__file__).parent / "config.yaml"
        if not fpath.exists():
            raise FileNotFoundError("Config file not found")
        else:
            self.config_fpath: Path = fpath
        obj_file_dir: Path = Path(__file__).parent / "assets" / "scene_objects"
        if not obj_file_dir.exists():
            raise FileNotFoundError("Scene objects directory not found")
        else:
            self.scene_objs_dir_path: Path = obj_file_dir
        self.all_scene_objs_fpaths: list = list(obj_file_dir.glob("*.obj"))

    def generate_video(self, sample_id, visualise: bool = False) -> None:
        # Load settings for given sample ID and generate video
        with open(self.config_fpath, "r") as f:
            settings_dict: dict = yaml.safe_load(f)
        if sample_id > len(settings_dict) - 1 or sample_id < 0:
            raise IndexError('Sample index out of bounds')
        render_params: dict = settings_dict[sample_id]["render_settings"]
        sim_params: dict = settings_dict[sample_id]["sim_settings"]

        if visualise:
            physicsClient = pb.connect(pb.GUI)
            pb.resetDebugVisualizerCamera(
                cameraDistance=5,
                cameraYaw=-30,
                cameraPitch=-52,
                cameraTargetPosition=[0, 5, 0],
            )
        else:
            physicsClient = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.807)

        scene_obj_ids = []
        for obj_fpath in tqdm(self.all_scene_objs_fpaths, desc="Loading scene objects"):
            obj_id = pb.createCollisionShape(
                shapeType=pb.GEOM_MESH, fileName=str(obj_fpath)
            )
            pb.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obj_id,
                baseOrientation=ROT_90_X_QUAT,
            )
            scene_obj_ids.append(obj_id)

        # Run sim
        n_steps: int = int(
            render_params.get("vid_length_mins")
            * render_params.get("fps")
            * sim_params.get("steps_per_frame")
        )
        for _ in range(n_steps):
            # print("Simulation start.")
            pb.stepSimulation()
            # print("Simulation end.")

        pb.disconnect()

    def generate_all_videos(self) -> None:
        with open(self.config_fpath, "r") as f:
            settings_dict: dict = yaml.safe_load(f)
        for i in range(len(settings_dict)):
            self.generate_video(i)
