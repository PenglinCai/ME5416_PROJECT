""" Rendering Script using POVray

This script reads simulated data file to render POVray animation movie.
The data file should contain dictionary of positions vectors, times, and radii.

The script supports multiple camera positions where a video is generated
for each camera view.

Notes
-----
    The module requires POVray installed.
"""

import multiprocessing
import os
import shutil
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import interpolate
from tqdm import tqdm

from _povmacros import Stages, pyelastica_rod, render

# Setup (USER DEFINE)
DATA_PATH = "./octopus_arm.dat"  # Path to the simulation data
SAVE_PICKLE = True

# Rendering Configuration (USER DEFINE)
OUTPUT_FILENAME = os.path.splitext(os.path.basename(DATA_PATH))[0]
OUTPUT_IMAGES_DIR = "frames"
OUTPUT_VIDEO_DIR = "video"
FPS = 90
WIDTH = 1920
HEIGHT = 1080
DISPLAY_FRAMES = "Off"  # ['On', 'Off']

# Camera/Light Configuration (USER DEFINE)
stages = Stages()
stages.add_camera(
    location=[3.0, 5, -12.0], angle=30, look_at=[0, 0, 2], name="diag",
)
stages.add_camera(
    location=[0, 8, 0], angle=0, look_at=[0, 0, 0], sky=[-1, 0, 0], name="top",
)
stages.add_light(position=[1500, 2500, -1000], color="White", camera_id=-1)
stages.add_light(position=[15.0, 10.5, -15.0], color=[0.09, 0.09, 0.1], camera_id=0)
stages.add_light(position=[0.0, 8.0, 5.0], color=[0.09, 0.09, 0.1], camera_id=1)
stage_scripts = stages.generate_scripts()

included = ["default.inc"]  # External POVray includes

# Multiprocessing Configuration (USER DEFINE)
MULTIPROCESSING = True
THREAD_PER_AGENT = 4
NUM_AGENT = multiprocessing.cpu_count() // 2


def load_simulation_data(path):
    assert os.path.exists(path), f"File does not exist: {path}"
    if SAVE_PICKLE:
        import pickle as pk
        with open(path, "rb") as fptr:
            return pk.load(fptr)
    else:
        raise NotImplementedError("Only pickled data is supported")


if __name__ == "__main__":
    # Clean up the old frames directory
    if os.path.exists(OUTPUT_IMAGES_DIR):
        shutil.rmtree(OUTPUT_IMAGES_DIR)

    # Create directories for each camera view
    for view_name in stage_scripts.keys():
        os.makedirs(os.path.join(OUTPUT_IMAGES_DIR, view_name), exist_ok=True)

    # Create the video output directory
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    # Load the simulation data
    data = load_simulation_data(DATA_PATH)

    # Convert data to numpy arrays
    time_array = np.array(data["time"])
    xs = np.array(data["position"])
    radii = np.array(data["radius"])

    # Interpolate data
    runtime = time_array.max()
    total_frame = int(runtime * FPS)
    times_true = np.linspace(0, runtime, total_frame)

    xs_interp = interpolate.interp1d(time_array, xs, axis=0)(times_true)
    radii_interp = interpolate.interp1d(time_array, radii, axis=0)(times_true)

    # Generate a batch of POVray scripts
    batch = []
    for frame_number in tqdm(range(total_frame), desc="Scripting"):
        for view_name, stage_script in stage_scripts.items():
            output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
            script = [f'#include "{s}"' for s in included] + [stage_script]

            # Use the interpolated radii for this frame
            r_frame = radii_interp[frame_number]

            # Add the soft rod object; r accepts the array of node radii
            rod_object = pyelastica_rod(
                x=xs_interp[frame_number],
                r=r_frame,
                color="rgb<0.45,0.39,1>"
            )
            script.append(rod_object)

            # Write the .pov file
            pov_script = "\n".join(script)
            file_base = os.path.join(output_path, f"frame_{frame_number:04d}")
            with open(file_base + ".pov", "w") as f:
                f.write(pov_script)
            batch.append(file_base)

    # Invoke POV-Ray to render frames
    pbar = tqdm(total=len(batch), desc="Rendering")
    if MULTIPROCESSING:
        func = partial(
            render,
            width=WIDTH,
            height=HEIGHT,
            display=DISPLAY_FRAMES,
            pov_thread=THREAD_PER_AGENT,
        )
        with Pool(NUM_AGENT) as p:
            for _ in p.imap_unordered(func, batch):
                pbar.update()
    else:
        for filename in batch:
            render(
                filename,
                width=WIDTH,
                height=HEIGHT,
                display=DISPLAY_FRAMES,
                pov_thread=os.cpu_count(),
            )
            pbar.update()

    # Use ffmpeg to compile frames into a video for each view
    for view_name in stage_scripts.keys():
        imageset_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
        video_name = f"{OUTPUT_FILENAME}_{view_name}.mp4"
        video_path = os.path.join(OUTPUT_VIDEO_DIR, video_name)
        os.system(
            f"ffmpeg -y -r {FPS} -i {imageset_path}/frame_%04d.png {video_path}"
        )
