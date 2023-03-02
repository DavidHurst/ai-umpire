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
        self.data_dir_path.mkdir(parents=True, exist_ok=True)

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

    def generate_video(
        self, sample_id, show_sim_gui: bool = False, show_render_gui: bool = False
    ) -> None:
        if sample_id > len(self.settings_dict) - 1 or sample_id < 0:
            raise IndexError("Sample index out of bounds")
        render_params: dict = self.settings_dict[sample_id]["render_settings"]
        sim_params: dict = self.settings_dict[sample_id]["sim_settings"]

        if show_sim_gui:
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
        pb.setRealTimeSimulation(0)
        nvisii.initialize(headless=not show_render_gui, lazy_updates=True)
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
        # Second index is in/out
        # Third index is up down
        camera.get_transform().look_at(at=(0, 0, 2.5), up=(0, 1, 0), eye=(0, -10, 2.5))
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

        pb_obj = pb.createCollisionShape(
            pb.GEOM_MESH, vertices=vertices, meshScale=scale,
        )
        pb.createMultiBody(
            baseCollisionShapeIndex=pb_obj, basePosition=pos, baseOrientation=rot,
        )

        ids_pybullet_and_nvisii_names = []
        for obj_fpath in tqdm(self.all_scene_objs_fpaths, desc="Loading scene objects"):
            if "metal" in str(obj_fpath) or 'line' in str(obj_fpath):
                continue

            pb_obj = pb.createCollisionShape(pb.GEOM_MESH, fileName=str(obj_fpath),)
            pb.createMultiBody(
                baseCollisionShapeIndex=pb_obj,
                baseMass=0,
                baseOrientation=ROT_90_X_QUAT,
            )
            pos, rot = pb.getBasePositionAndOrientation(pb_obj)

            nvisii_name: str = str(obj_fpath).split("/")[-1][:-4]
            print(f"Loading {obj_fpath}")
            nvisii_obj = nvisii.entity.create(
                name=nvisii_name,
                transform=nvisii.transform.create(nvisii_name),
                material=nvisii.material.create(nvisii_name),
            )
            nvisii_obj.set_mesh(
                nvisii.mesh_create_from_file(nvisii_name, str(obj_fpath))
            )
            nvisii_obj.get_transform().set_position(pos)
            nvisii_obj.get_transform().set_rotation(rot)

            rgb = colorsys.hsv_to_rgb(
                random.uniform(0, 1), random.uniform(0.7, 1), random.uniform(0.7, 1)
            )
            nvisii_obj.get_material().set_base_color(rgb)
            obj_mat = nvisii_obj.get_material()
            obj_mat.set_metallic(0)  # should 0 or 1
            obj_mat.set_transmission(random.uniform(0.9, 1))  # should 0 or 1

            ids_pybullet_and_nvisii_names.append(
                {"pybullet_id": pb_obj, "nvisii_id": nvisii_name}
            )

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
        for i in range(10):
            print('i=',i)
            for _ in range(sim_params["steps_per_frame"]):
                pb.stepSimulation()

            # Update mesh positions
            for ids in ids_pybullet_and_nvisii_names:
                pos, rot = pb.getBasePositionAndOrientation(ids["pybullet_id"])

                nvisii_obj = nvisii.entity.get(ids["nvisii_id"])
                nvisii_obj.get_transform().set_position(pos)
                nvisii_obj.get_transform().set_rotation(
                    rot
                )  # nvisii quat expects w as the first argument
                print(f"Rendering frame {str(i).zfill(5)}/{str(n_frames).zfill(5)}")
                print('i=', i)
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
                f"sample{sample_id}.mp4",
            ],
            cwd=self.data_dir_path.resolve(),
        )

    def generate_all_videos(self) -> None:
        with open(self.config_fpath, "r") as f:
            settings_dict: dict = yaml.safe_load(f)
        for i in range(len(settings_dict)):
            self.generate_video(i)
