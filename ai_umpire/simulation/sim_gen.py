__all__ = ["SimGenerator"]

import glob
from pathlib import Path

import cv2


class SimGenerator:
    def __init__(self, frames_file: Path):
        self._frames: Path = frames_file

    def convert_frames_to_vid(self, out_file: Path) -> None:
        img_array: list = []
        size: tuple = ()
        for filename in glob.glob(f"{self._frames}/*.png"):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        writer = cv2.VideoWriter(
            filename='sim0.mp4',
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=50,
            frameSize=size,
        )

        for i in range(len(img_array)):
            print(f"Writing number: {i}")
            writer.write(img_array[i])
        writer.release()
