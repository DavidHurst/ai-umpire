import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

root_dir = os.getcwd()
frames_dir = root_dir + '\\prototyping\\players_sim'
img_paths = [f'{frames_dir}\\{filename}' for filename in os.listdir(frames_dir)]

img_ = cv2.imread(img_paths[0], 1)
averaged_frames = np.zeros_like(img_, float)

n_frames_to_average = 20
i = 0
im_count = 0
for path in img_paths:
    img = cv2.imread(path, 1)
    img_rbg = img[..., ::-1].copy()
    imArr = np.array(img_rbg, dtype=float)
    averaged_frames += imArr / n_frames_to_average
    i += 1

    if i == n_frames_to_average:
        blurred = np.array(np.round(averaged_frames), dtype=np.uint8)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        fname = root_dir + '\\players_sim_blurred\\' + f'blurred{im_count}.jpg'
        cv2.imwrite(fname, blurred)

        averaged_frames = np.zeros_like(img_, float)
        print(f'Saved image #{im_count} to {fname}')
        i = 0
        im_count += 1

exit()
n_frames_to_average = 10
averaged_frames = np.zeros_like(img_, float)
i = 0
for img_name in imlist:
    imArr = np.array(cv2.imread(os.path.join(frames_dir, img_name)), dtype=float)
    averaged_frames += imArr / n_frames_to_average
    i += 1

    if i == n_frames_to_average:
        print(img_name + '100fps')
        blurred = np.array(np.round(averaged_frames), dtype=np.uint8)
        # blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)     # Swap channels and look into what cv2 expectsas input for imwrite

        cv2.imwrite(root_dir + '/100fps/' + f'/{img_name[:-4]}.jpg', blurred)
        plt.plot(blurred)
        plt.show()

        averaged_frames = np.zeros_like(img_, float)
        i = 0
