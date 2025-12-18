from ml_tools.path_manager import DragonPathManager

# 1. Initialize the PathManager using this file as the anchor.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["resources"]
)

# 2. Define and register specific file paths.
PM.model_weights_three = PM.resources / "AlloysDendritesSpheroids.pth"
PM.model_weights_two = PM.resources / "DendritesSpheroids.pth"
PM.sam_weights_file = PM.resources / "sam2_1_hiera_large.pth"

# 3. Check status
if __name__ == "__main__":
    PM.status()
