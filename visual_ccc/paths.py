from ml_tools.path_manager import DragonPathManager

# 1. Initialize the PathManager using this file as the anchor.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["resources"]
)

# 2. Define and register specific file paths.
PM.model_weights = PM.resources / "model_weights.pth"

# 3. Check status
if __name__ == "__main__":
    PM.status()
