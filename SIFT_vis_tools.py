import matplotlib.pyplot as plt
from matplotlib.patches import Circle
    

def visualize_pyramid(self, pyramid):
    fig, axes = plt.subplots(
        nrows=len(pyramid), ncols=len(pyramid[0]), figsize=(12, 12)
    )

    for i in range(len(pyramid)):
        for j in range(len(pyramid[i])):
            axes[i, j].imshow(pyramid[i][j], cmap="gray")
            axes[i, j].set_title(f"Octave {i}, Image {j}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_DOC_for_octave(self, DOG):
    fig, axes = plt.subplots(nrows=len(DOG), ncols=len(DOG[0]), figsize=(12, 12))

    for i in range(len(DOG)):
        for j in range(len(DOG[i])):
            axes[i, j].imshow(DOG[i][j], cmap="gray")
            axes[i, j].set_title(f"Octave {i}, Image {j}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_keypoints(self, pyramid, keypoints):
    fig, axes = plt.subplots(
        nrows=len(pyramid), ncols=len(pyramid[0]), figsize=(12, 12)
    )

    for i in range(len(pyramid)):
        for j in range(len(pyramid[i])):
            axes[i, j].imshow(pyramid[i][j], cmap="gray")
            axes[i, j].set_title(f"Octave {i}, Image {j}")
            axes[i, j].axis("off")
            for kp in keypoints[i]:
                x = kp[0]
                y = kp[1]
                circle = Circle((x, y), radius=2, color="r", fill=True)
                axes[i, j].add_patch(circle)
    plt.tight_layout()
    plt.show()
