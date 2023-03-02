__all__ = ["VideoGenerator"]

import colorsys
import os
import subprocess
from pathlib import Path
import random

import nvisii
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

        with open(self.config_fpath, "r") as f:
            self.settings_dict: dict = yaml.safe_load(f)

    def generate_video(self, sample_id, visualise: bool = False) -> None:
        if sample_id > len(self.settings_dict) - 1 or sample_id < 0:
            raise IndexError("Sample index out of bounds")
        render_params: dict = self.settings_dict[sample_id]["render_settings"]
        sim_params: dict = self.settings_dict[sample_id]["sim_settings"]

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
        nvisii.initialize(headless=not visualise, lazy_updates=True)
        if render_params["enable_denoiser"]:
            nvisii.enable_denoiser()

        camera = nvisii.entity.create(
            name="camera",
            transform=nvisii.transform.create("camera"),
            camera=nvisii.camera.create_from_fov(
                name="camera",
                field_of_view=0.85,
                aspect=float(render_params["width"]) / float(render_params["height"]),
            ),
        )

        camera.get_transform().look_at(
            at=(0, 0, 0), up=(0, 0, 1), eye=(10, 0, 4),
        )
        nvisii.set_camera_entity(camera)
        nvisii.set_dome_light_intensity(1.0)
        nvisii.set_dome_light_sky(
            sun_position=(10, 10, 10), atmosphere_thickness=1.0, saturation=1.0
        )
        sun = nvisii.entity.create(
            name="sun",
            mesh=nvisii.mesh.create_sphere("sphere"),
            transform=nvisii.transform.create("sun"),
            light=nvisii.light.create("sun"),
        )
        sun.get_transform().set_position((10, 10, 10))
        sun.get_light().set_temperature(5780)
        sun.get_light().set_intensity(1000)
        floor = nvisii.entity.create(
            name="floor",
            mesh=nvisii.mesh.create_plane("floor"),
            transform=nvisii.transform.create("floor"),
            material=nvisii.material.create("floor"),
        )
        floor.get_transform().set_position((0, 0, 0))
        floor.get_transform().set_scale((10, 10, 10))
        floor.get_material().set_roughness(0.1)
        floor.get_material().set_base_color((0.5, 0.5, 0.5))

        vertices = floor.get_mesh().get_vertices()
        pos = floor.get_transform().get_position()
        pos = [pos[0], pos[1], pos[2]]
        scale = floor.get_transform().get_scale()
        scale = [scale[0], scale[1], scale[2]]
        rot = floor.get_transform().get_rotation()
        rot = [rot[0], rot[1], rot[2], rot[3]]

        obj_col_id = pb.createCollisionShape(
            pb.GEOM_MESH, vertices=vertices, meshScale=scale,
        )
        pb.createMultiBody(
            baseCollisionShapeIndex=obj_col_id, basePosition=pos, baseOrientation=rot,
        )

        mesh = nvisii.mesh.create_teapotahedron("mesh")

        # set up for pybullet - here we will use indices for
        # objects with holes
        vertices = mesh.get_vertices()
        indices = mesh.get_triangle_indices()

        seconds_per_step: float = 1.0 / 240.0
        ids_pybullet_and_nvisii_names = []
        name = "mesh_0"
        obj = nvisii.entity.create(
            name=name,
            transform=nvisii.transform.create(name),
            material=nvisii.material.create(name),
        )
        obj.set_mesh(mesh)

        # transforms
        pos = nvisii.vec3(
            random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(2, 5)
        )
        rot = nvisii.normalize(
            nvisii.quat(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            )
        )
        s = random.uniform(0.2, 0.5)
        scale = (s, s, s)

        obj.get_transform().set_position(pos)
        obj.get_transform().set_rotation(rot)
        obj.get_transform().set_scale(scale)

        # pybullet setup
        pos = [pos[0], pos[1], pos[2]]
        rot = [rot[0], rot[1], rot[2], rot[3]]
        scale = [scale[0], scale[1], scale[2]]

        obj_col_id = pb.createCollisionShape(
            pb.GEOM_MESH, vertices=vertices, meshScale=scale,
        )
        pb.createMultiBody(
            baseCollisionShapeIndex=obj_col_id,
            basePosition=pos,
            baseOrientation=rot,
            baseMass=random.uniform(0.5, 2),
        )

        rgb = colorsys.hsv_to_rgb(
            random.uniform(0, 1), random.uniform(0.7, 1), random.uniform(0.7, 1)
        )

        obj.get_material().set_base_color(rgb)

        obj_mat = obj.get_material()
        obj_mat.set_metallic(0)  # should 0 or 1
        obj_mat.set_transmission(random.uniform(0.9, 1))  # should 0 or 1

        # scene_obj_ids = []
        # for obj_fpath in tqdm(self.all_scene_objs_fpaths, desc="Loading scene objects"):
        #     obj_id = pb.createCollisionShape(
        #         shapeType=pb.GEOM_MESH, fileName=str(obj_fpath)
        #     )
        #     pb.createMultiBody(
        #         baseMass=0,
        #         baseCollisionShapeIndex=obj_id,
        #         baseOrientation=ROT_90_X_QUAT,
        #     )
        #     scene_obj_ids.append(obj_id)

        # Run sim
        n_steps: int = int(
            render_params["vid_length_mins"]
            * render_params["fps"]
            * sim_params["steps_per_frame"]
        )
        n_frames: int = int(
            render_params["vid_length_mins"] * 60 * render_params["fps"]
        )
        print(f"Running sim for {n_steps} steps, {n_frames} frames")
        for i in range(2):
            for _ in range(sim_params["steps_per_frame"]):
                pb.stepSimulation()
            pos, rot = pb.getBasePositionAndOrientation(obj_col_id)

            # get the nvisii entity for that object
            obj_entity = nvisii.entity.get(name)
            obj_entity.get_transform().set_position(pos)

            # nvisii quat expects w as the first argument
            obj_entity.get_transform().set_rotation(rot)
            print(f"rendering frame {str(i).zfill(5)}/{str(n_frames).zfill(5)}")
            nvisii.render_to_file(
                width=int(render_params["width"]),
                height=int(render_params["height"]),
                samples_per_pixel=int(render_params["spp"]),
                file_path=str(self.data_dir_path / f"{str(i).zfill(5)}.png"),
            )
        pb.disconnect()
        nvisii.deinitialize()
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(render_params["fps"]),
                "-i",
                r"%05d.png",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                f'sample{sample_id}.mp4',
            ],
            cwd=self.data_dir_path.resolve(),
        )

    def generate_all_videos(self) -> None:
        with open(self.config_fpath, "r") as f:
            settings_dict: dict = yaml.safe_load(f)
        for i in range(len(settings_dict)):
            self.generate_video(i)
