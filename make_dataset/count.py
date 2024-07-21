import glob
import os

path = os.path.join('/mnt/HDD10TB-1/sugiura/HRNet-Human-Pose-Estimation/make_dataset/num_keypoints_image/pickle', '*', '*', '*')

a = glob.glob(path)

print(len(a))
