import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

frames_dir = "./ai-umpire/frames"
root_dir = './ai-umpire'

imlist = [filename for filename in os.listdir(frames_dir) if filename[-4:] in [".jpg", ".JPG"]]

img_ = cv2.imread(os.path.join(frames_dir, imlist[0]))

averaged_frames = np.zeros_like(img_, float)

n_frames_to_average = 20
i = 0
for img_name in imlist:
    imArr = np.array(cv2.imread(os.path.join(frames_dir, img_name)), dtype=float)
    averaged_frames += imArr / n_frames_to_average
    i += 1

    if i == n_frames_to_average:
        print(img_name + '50fps')
        blurred = np.array(np.round(averaged_frames), dtype=np.uint8)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        cv2.imwrite(root_dir + '/50fps/' + f'/{img_name[:-4]}.jpg', blurred)

        averaged_frames = np.zeros_like(img_, float)
        i = 0

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
