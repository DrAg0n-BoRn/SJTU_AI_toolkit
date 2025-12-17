from ml_tools.path_manager import DragonPathManager

# 1. Initialize the PathManager using this file as the anchor.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["results", "backups"]
)

# 2. Define and register specific paths.
PM.two_dataset = PM.ROOT / "two_classes"
PM.three_dataset = PM.ROOT / "three_classes"
PM.artifacts = PM.results / "artifacts"
PM.checkpoints = PM.results / "checkpoints"
PM.metrics = PM.results / "metrics"

PM.sam_inputs = PM.ROOT / "sam_inputs"
PM.sam_outputs = PM.ROOT / "sam_outputs"
PM.sam_artifacts = PM.ROOT / "sam_artifacts"

PM.sam_weights_file = PM.sam_artifacts / "sam2_1_hiera_large.pth"
PM.transform_recipe_file = PM.artifacts / "transform_recipe.json"

# 3. Check status
if __name__ == "__main__":
    PM.make_dirs()
    PM.status()
