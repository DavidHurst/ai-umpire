from pathlib import Path

from ai_umpire import SimVideoGen, Localiser

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 2
sim_frames_path: Path = root_dir_path / "sim_frames" / f"sim_{sim_id}_frames"
sim_blurred_frames_path: Path = (
    root_dir_path / "blurred_frames" / f"sim_{sim_id}_blurred"
)
vid_dir: Path = root_dir_path / "videos"

# Generate video from simulation frames if it does not already exist
if not (vid_dir / f"vid_{sim_id}.mp4").exists():
    print(f"Generating video for sim id {sim_id}")
    vid_gen = SimVideoGen(root_dir=root_dir_path)
    vid_gen.convert_frames_to_vid(sim_id, 50)

# Generate ball candidates per frame in video
loc = Localiser()
candidates = loc.get_ball_candidates(vid_dir / f"sim_{sim_id}.mp4")
