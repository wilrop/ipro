import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import imageio.v3 as iio


def plot_patches(patches, pf, points, log_dir, name):
    pf = sorted(pf, key=lambda x: x[0])
    fig, ax = plt.subplots()
    ax.set_facecolor((154 / 256, 142 / 256, 148 / 256, 0.8))
    ax.scatter(points[:, 0], points[:, 1], c='b', s=15)

    rectangles = [(tuple(patch.bot_left), patch.width, patch.height) for patch in patches]

    for rect in rectangles:
        ax.add_patch(Rectangle(rect[0], rect[1], rect[2], linewidth=1, edgecolor='r', facecolor='white'))

    for point1, point2 in zip(pf, pf[1:]):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], c='black', linewidth=1)

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{name}.png")
    plt.savefig(file_path)


def create_gif(log_dir, name):
    images = []
    for file_name in sorted(os.listdir(log_dir)):
        if file_name.startswith('patches'):
            file_path = os.path.join(log_dir, file_name)
            images.append(iio.imread(file_path))
    iio.imwrite(os.path.join(log_dir, f"{name}.gif"), images, format='GIF', duration=1000)
