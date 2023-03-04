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
        self.assets_dir_path: Path = Path(__file__).parent / "assets"
        self.obj_file_dir: Path = self.assets_dir_path / "scene_objects"
        if not self.obj_file_dir.exists():
            raise FileNotFoundError("Scene objects directory not found")
        else:
            self.scene_objs_dir_path: Path = self.obj_file_dir
        self.all_scene_objs_fpaths: list = list(self.obj_file_dir.glob("*.obj"))

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

        # Init pybullet
        if show_sim_gui:
            physicsClient = pb.connect(pb.GUI)
            pb.resetDebugVisualizerCamera(
                cameraDistance=15,
                cameraYaw=0,
                cameraPitch=-10,
                cameraTargetPosition=[0, 2, 5],
            )
        else:
            physicsClient = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.807)
        pb.setRealTimeSimulation(0)

        # Init nvisii
        nvisii.initialize(headless=not show_render_gui, lazy_updates=True)
        self.create_materials(render_params["court_colour"])
        if render_params["enable_denoiser"]:
            nvisii.enable_denoiser()
        nvisii.configure_denoiser(use_albedo_guide=False, use_normal_guide=False)

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
        camera.get_transform().look_at(
            at=(0, 0, 2.5), up=(0, 1, 0), eye=(0, -9.25, 2.5)
        )
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
                mesh=nvisii.mesh.create_box(f"box_{i}", (0.4, 0.8, 0.1)),
            )
            for i in range(10)
        ]
        for light in ceiling_lights:
            light.get_light().set_temperature(6000)
            light.get_light().set_intensity(5000)
            light.get_light().set_falloff(2)
        ceiling_lights[0].get_transform().set_position((-3, -1, 8))
        ceiling_lights[1].get_transform().set_position((-3, 4, 8))
        ceiling_lights[2].get_transform().set_position((-3, 7, 8))
        ceiling_lights[3].get_transform().set_position((-3, 10, 8))
        ceiling_lights[4].get_transform().set_position((-3, -2, 8))

        ceiling_lights[5].get_transform().set_position((3, -1, 8))
        ceiling_lights[6].get_transform().set_position((3, 4, 8))
        ceiling_lights[7].get_transform().set_position((3, 7, 8))
        ceiling_lights[8].get_transform().set_position((3, 10, 8))
        ceiling_lights[9].get_transform().set_position((3, -2, 8))

        crowd_lights: list = [
            nvisii.entity.create(
                name=f"crowd_light_{i}",
                transform=nvisii.transform.create(f"crowd_light_{i}"),
                light=nvisii.light.create(f"crowd_light_{i}"),
            )
            for i in range(3)
        ]
        for light in crowd_lights:
            light.get_light().set_temperature(5000)
            light.get_light().set_intensity(40)
            light.get_light().set_falloff(2)
        crowd_lights[0].get_transform().set_position((-6, 0, 6))
        crowd_lights[1].get_transform().set_position((6, 0, 6))
        crowd_lights[2].get_transform().set_position((0, -9, 6))

        pybullet_and_nvisii_ids = []
        for obj_fpath in self.all_scene_objs_fpaths:  # , desc="Loading scene objects"):
            nvisii_obj_id: str = str(obj_fpath).split("/")[-1][:-4]
            if "spectator" in nvisii_obj_id and random.uniform(0, 1) > (
                1 - render_params["crowd_fullness"]
            ):
                continue
            if "wall" in nvisii_obj_id:  # Only create PB objects for court walls
                pb_obj_id = pb.createCollisionShape(
                    pb.GEOM_MESH, fileName=str(obj_fpath),
                )
                pb.createMultiBody(
                    baseCollisionShapeIndex=pb_obj_id,
                    baseMass=0,
                    baseOrientation=ROT_90_X_QUAT,  # Correct for blenders different axis
                )
                nvisii_obj = nvisii.entity.create(
                    name=nvisii_obj_id,
                    transform=nvisii.transform.create(nvisii_obj_id),
                    material=nvisii.material.create(nvisii_obj_id),
                )
                nvisii_obj.set_mesh(
                    nvisii.mesh_create_from_file(nvisii_obj_id, str(obj_fpath))
                )

                pybullet_and_nvisii_ids.append(
                    {"pybullet_id": pb_obj_id, "nvisii_id": nvisii_obj_id}
                )
            else:
                nvisii_obj = nvisii.entity.create(
                    name=nvisii_obj_id,
                    transform=nvisii.transform.create(nvisii_obj_id),
                    material=nvisii.material.create(nvisii_obj_id),
                )
                nvisii_obj.set_mesh(
                    nvisii.mesh_create_from_file(nvisii_obj_id, str(obj_fpath))
                )
            nvisii_obj.get_transform().set_rotation(
                ROT_90_X_QUAT
            )  # Correct for blenders different axis

            # Apply materials to objects
            if "glass" in nvisii_obj_id:
                nvisii_obj.set_material(nvisii.material.get("glass_mat"))
            elif "metal" in nvisii_obj_id:
                nvisii_obj.set_material(nvisii.material.get("metal_mat"))
            elif "line" in nvisii_obj_id:
                nvisii_obj.set_material(nvisii.material.get("court_markings_mat"))
            elif "spectator" in nvisii_obj_id or "photographer" in nvisii_obj_id:
                nvisii_obj.set_material(nvisii.material.get("spectator_mat"))
            elif "text" in nvisii_obj_id:
                nvisii_obj.set_material(nvisii.material.get("text_mat"))
            elif "wall" in nvisii_obj_id:
                if "tin" in nvisii_obj_id:
                    if "line" in nvisii_obj_id:
                        nvisii_obj.set_material(
                            nvisii.material.get("court_marking_mat")
                        )
                    else:
                        nvisii_obj.set_material(nvisii.material.get("dark_matte_mat"))
                elif "floor" in nvisii_obj_id:
                    nvisii_obj.set_material(nvisii.material.get("court_floor_mat"))
                else:
                    if "back" in nvisii_obj_id:
                        nvisii_obj.set_material(nvisii.material.get("glass_mat"))
                    else:
                        nvisii_obj.set_material(nvisii.material.get("court_glass_mat"))
            else:
                nvisii_obj.set_material(nvisii.material.get("dark_matte_mat"))

        # Load squash ball
        pb_sph_obj = pb.loadURDF(
            str(self.assets_dir_path / "sphere_with_restitution.urdf"), sim_params['init_ball_pos']
        )
        nvisii_sph_obj = nvisii.entity.create(
            name="ball",
            transform=nvisii.transform.create("ball"),
            material=nvisii.material.create("ball"),
        )
        nvisii_sph_obj.set_mesh(
            nvisii.mesh_create_from_file(
                "ball", str(self.assets_dir_path / "textured_sphere_smooth.obj")
            )
        )
        pybullet_and_nvisii_ids.append({"pybullet_id": pb_sph_obj, "nvisii_id": "ball"})
        pos, rot = pb.getBasePositionAndOrientation(pb_sph_obj)
        nvisii_sph_obj.get_transform().set_position(pos)
        nvisii_sph_obj.get_transform().set_rotation(rot)

        # Run sim
        n_steps: int = int(
            render_params["vid_length_secs"]
            * render_params["fps"]
            * sim_params["steps_per_frame"]
        )
        n_frames: int = int(render_params["vid_length_secs"] * render_params["fps"])
        print(f"Running sim for {n_steps} steps, {n_frames} frames")
        for i in range(n_frames):
            for k in range(sim_params["steps_per_frame"]):
                pb.stepSimulation()

            # Update mesh positions
            for ids in pybullet_and_nvisii_ids:
                pos, rot = pb.getBasePositionAndOrientation(ids["pybullet_id"])
                nvisii_obj = nvisii.entity.get(ids["nvisii_id"])
                nvisii_obj.get_transform().set_position(pos)
                nvisii_obj.get_transform().set_rotation(rot)

            print(f"Rendering frame {str(i).zfill(5)}/{str(n_frames).zfill(5)}")
            pos, _ = pb.getBasePositionAndOrientation(pb_sph_obj)
            print(f"Ball @ {pos}")
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

    def create_materials(self, court_colour: str):
        if court_colour == "blue":
            nvisii.material = nvisii.material.create(
                "court_markings_mat",
                (0.31, 0.54, 0.99),
                specular=0,
                sheen_tint=0.5,
                roughness=0.3,
                clearcoat=0,
                clearcoat_roughness=0.03,
                ior=1.45,
            )
            nvisii.material = nvisii.material.create(
                "court_glass_mat",
                (0.2, 0.6, 1),
                transmission=0.89,
                transmission_roughness=0,
                ior=1,
                specular=0,
                metallic=0.2,
            )
        else:
            raise NotImplemented(f"Colour [{court_colour}] not implemented")
        nvisii.material.create(
            "glass_mat",
            (1, 1, 1),
            transmission=1,
            transmission_roughness=0,
            ior=1,
            specular=0,
            metallic=0.2,
        )
        nvisii.material.create("metal_mat", (0.9, 0.9, 0.9), metallic=1)
        nvisii.material.create(
            "dark_matte_mat", (0.001, 0.001, 0.001), roughness=0.8, specular=0
        )
        nvisii.material.create("spectator_mat", (0.2, 0.2, 0.2), specular=0.3)
        court_floor_mat = nvisii.material.create("court_floor_mat", (0.29, 0.25, 0.19))
        wood_tex: nvisii.texture = nvisii.texture.create_from_file(
            "wood_tex", str((self.assets_dir_path / "wood_texture.jpg").resolve())
        )
        court_floor_mat.set_base_color_texture(wood_tex)
        nvisii.material.create("text_mat", (1, 1, 1))
