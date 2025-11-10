from ml_tools.path_manager import DragonPathManager

# 1. Initialize the PathManager using this file as the anchor.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["results", "backups", "dataset"]
)

# 2. Define and register specific file paths.
PM.artifacts = PM.results / "artifacts"
PM.checkpoints = PM.results / "checkpoints"
PM.metrics = PM.results / "metrics"
PM.validation = PM.results / "validation"
PM.original_images = PM.ROOT / "Original_Images"

PM.transform_recipe_file = PM.artifacts / "transform_recipe.json"

# 3. Check status
if __name__ == "__main__":
    PM.make_dirs()
    PM.status()
