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

        out_dir: Path = (self.data_dir_path / f"sample_{sample_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

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

        ceiling_lights: list = [
            nvisii.entity.create(
                name=f"ceiling_light_{i}",
                transform=nvisii.transform.create(f"ceiling_light_{i}"),
                light=nvisii.light.create(f"ceiling_light_{i}"),
                # mesh=nvisii.mesh.create_box(f"sphere_{i}"),
            )
            for i in range(6)
        ]
        for light in ceiling_lights:
            light.get_light().set_temperature(6000)
            light.get_light().set_intensity(150)
            light.get_light().set_falloff(2)
        ceiling_lights[0].get_transform().set_position((-3, 2, 8))
        ceiling_lights[1].get_transform().set_position((-3, 5, 8))
        ceiling_lights[2].get_transform().set_position((-4, 8, 8))
        ceiling_lights[3].get_transform().set_position((3, 2, 8))
        ceiling_lights[4].get_transform().set_position((3, 5, 8))
        ceiling_lights[5].get_transform().set_position((3, 8, 8))

        # wall_glass = nvisii.material.create('wall_glass', (0.2, 0.5, 0.8), 0.3, 0, 0.5, transmission=0.8)

        floor = nvisii.entity.create(
            name="floor",
            mesh=nvisii.mesh.create_plane("floor"),
            transform=nvisii.transform.create("floor"),
            material=nvisii.material.create("floor"),
        )
        floor.get_transform().set_position((0, 0, 0))
        floor.get_transform().set_scale((50, 50, 50))
        floor.get_material().set_roughness(0.1)
        floor.get_material().set_base_color((0.2, 0.2, 0.2))

        backdrop = nvisii.entity.create(
            name="backdrop",
            mesh=nvisii.mesh.create_plane("backdrop"),
            transform=nvisii.transform.create("backdrop"),
            material=nvisii.material.create("backdrop"),
        )
        backdrop.get_transform().set_position((0, 20, 5))
        backdrop.get_transform().set_scale((50, 50, 50))
        backdrop.get_transform().set_rotation((0.7071068, 0, 0, 0.7071068))
        backdrop.get_material().set_roughness(0.1)
        backdrop.get_material().set_base_color((0.2, 0.2, 0.2))

        ids_pybullet_and_nvisii_names = []
        for obj_fpath in tqdm(self.all_scene_objs_fpaths, desc="Loading scene objects"):
            pb_obj_id = pb.createCollisionShape(pb.GEOM_MESH, fileName=str(obj_fpath),)
            pb.createMultiBody(
                baseCollisionShapeIndex=pb_obj_id,
                baseMass=0,
                baseOrientation=ROT_90_X_QUAT,
            )
            pos, rot = pb.getBasePositionAndOrientation(pb_obj_id)

            nvisii_obj_id: str = str(obj_fpath).split("/")[-1][:-4]
            nvisii_obj = nvisii.entity.create(
                name=nvisii_obj_id,
                transform=nvisii.transform.create(nvisii_obj_id),
                material=nvisii.material.create(nvisii_obj_id),
            )
            nvisii_obj.set_mesh(
                nvisii.mesh_create_from_file(nvisii_obj_id, str(obj_fpath))
            )
            nvisii_obj.get_transform().set_position(pos)
            nvisii_obj.get_transform().set_rotation(rot)
            # create a material for each material you want and just set it for each obj,
            # based on blender principled shader

            if "glass" in str(obj_fpath) or "wall" in str(obj_fpath):
                if "back" not in str(obj_fpath) and "glass" not in str(obj_fpath):
                    nvisii_obj.get_material().set_base_color((0.2, 0.5, 0.8))
                obj_mat = nvisii_obj.get_material()
                obj_mat.set_metallic(0)
                obj_mat.set_transmission(1)
            if "line" in str(obj_fpath):
                nvisii_obj.get_material().set_base_color((0.3, 0.24, 0.9))
            if "metal" in str(obj_fpath):
                obj_mat = nvisii_obj.get_material()
                obj_mat.set_metallic(1)
                obj_mat.set_specular(7)
                nvisii_obj.get_material().set_base_color((0.8, 0.8, 0.8))

            ids_pybullet_and_nvisii_names.append(
                {"pybullet_id": pb_obj_id, "nvisii_id": nvisii_obj_id}
            )

        # Run sim
        n_steps: int = int(
            render_params["vid_length_secs"]
            * render_params["fps"]
            * sim_params["steps_per_frame"]
        )
        n_frames: int = int(
            render_params["vid_length_secs"] * render_params["fps"]
        )
        print(f"Running sim for {n_steps} steps, {n_frames} frames")
        for i in range(n_frames):
            for _ in range(sim_params["steps_per_frame"]):
                pb.stepSimulation()

            # Update mesh positions
            for ids in ids_pybullet_and_nvisii_names:
                pos, rot = pb.getBasePositionAndOrientation(ids["pybullet_id"])
                nvisii_obj = nvisii.entity.get(ids["nvisii_id"])
                nvisii_obj.get_transform().set_position(pos)
                nvisii_obj.get_transform().set_rotation(rot)

            print(f"Rendering frame {str(i).zfill(5)}/{str(n_frames).zfill(5)}")
            nvisii.render_to_file(
                width=int(render_params["width"]),
                height=int(render_params["height"]),
                samples_per_pixel=int(render_params["spp"]),
                file_path=str(out_dir / f"frame{str(i).zfill(5)}.png"),
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
                r"frame%05d.png",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                f"video_sample_{str(sample_id).zfill(2)}.mp4",
            ],
            cwd=out_dir.resolve(),
        )

    def generate_all_videos(self) -> None:
        with open(self.config_fpath, "r") as f:
            settings_dict: dict = yaml.safe_load(f)
        for i in range(len(settings_dict)):
            self.generate_video(i)
