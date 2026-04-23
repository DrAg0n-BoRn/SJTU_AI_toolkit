---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: visual-ccc
    language: python
    name: python3
---

```python
from torchvision import transforms
from torch.optim import AdamW
from typing import Literal

from ml_tools.ML_datasetmaster import DragonDatasetVision
from ml_tools.ML_trainer import DragonTrainer
from ml_tools.ML_callbacks import DragonModelCheckpoint, DragonPatienceEarlyStopping, DragonPlateauScheduler
from ml_tools.ML_utilities import inspect_model_architecture
from ml_tools.ML_configuration import (FormatBinaryImageClassificationMetrics, 
                                       FinalizeBinaryImageClassification,
                                       FormatMultiClassImageClassificationMetrics, 
                                       FinalizeMultiClassImageClassification,
                                       DragonTrainingConfig)
from ml_tools.IO_tools import train_logger
from ml_tools.keys import TaskKeys

from rootpaths import PM
from visual_ccc.gradcam import custom_alexnet, SIZE_REQUIREMENT
from visual_ccc.visualcnn_model import VisualCNN
```

```python
# Select model and number of classes
CLASSES: Literal["2-class", "3-class"] = "3-class"
MODEL: Literal["visualcnn", "alexnet"] = "visualcnn"
```

```python
if CLASSES == "2-class": # type: ignore
    vision_dataset = DragonDatasetVision.from_folder(PM.two_dataset)
    TASK = TaskKeys.BINARY_IMAGE_CLASSIFICATION
    if MODEL == "alexnet": # type: ignore
        pth_name = "DendritesSpheroids"
    else:
        pth_name = "V-DendritesSpheroids"
else:
    vision_dataset = DragonDatasetVision.from_folder(PM.three_dataset)
    TASK = TaskKeys.MULTICLASS_IMAGE_CLASSIFICATION
    if MODEL == "alexnet": # type: ignore
        pth_name = "AlloysDendritesSpheroids"
    else:
        pth_name = "V-AlloysDendritesSpheroids"


train_config = DragonTrainingConfig(validation_size=0.2,
                                    test_size=0.1,
                                    initial_learning_rate=0.0001,
                                    batch_size=2,
                                    task=TASK,
                                    device="cuda:0",
                                    finalized_filename=pth_name,
                                    random_state=101,
                                    
                                    early_stop_patience=30,
                                    scheduler_patience=4,
                                    scheduler_lr_factor=0.5,
                                    classes=CLASSES,
                                    model=MODEL)
```

## Binary classification: Dendrites, Spheroids

```python
vision_dataset.split_data(val_size=train_config.validation_size, 
                          test_size=train_config.test_size,
                          random_state=train_config.random_state)

vision_dataset.configure_transforms(resize_size=int(1.2*SIZE_REQUIREMENT),
                                    crop_size=SIZE_REQUIREMENT,
                                    mean=None, std=None,
                                    pre_transforms=[transforms.Grayscale(num_output_channels=1)])
```

```python
class_map = vision_dataset.save_class_map(save_dir=PM.artifacts)

vision_dataset.save_transform_recipe(filepath=PM.transform_recipe_file)
```

```python
# Model
if train_config.model == "alexnet": # type: ignore
    model = custom_alexnet(classes=train_config.classes)
else:
    model = VisualCNN(classes=train_config.classes)

inspect_model_architecture(model=model, save_dir=PM.artifacts)

# Optimizer
optimizer = AdamW(params=model.parameters(), lr=train_config.initial_learning_rate)

# Trainer
trainer = DragonTrainer(model=model,
                    train_dataset=vision_dataset.train_dataset,
                    validation_dataset=vision_dataset.validation_dataset,
                    kind=train_config.task,
                    optimizer=optimizer,
                    device=train_config.device,
                    checkpoint_callback=DragonModelCheckpoint(save_dir=PM.checkpoints, mode="min"),
                    early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, mode="min"),
                    lr_scheduler_callback=DragonPlateauScheduler(mode="min",
                                                                factor=train_config.scheduler_lr_factor,
                                                                patience=train_config.scheduler_patience)
                    )
```

```python
history = trainer.fit(save_dir=PM.artifacts, epochs=200, batch_size=train_config.batch_size)
```

```python
train_logger(train_config=train_config,
             model_parameters={"available models": ["visualcnn", "alexnet"]},
             train_history=history,
             save_directory=PM.results)
```

```python
# Configurations
if train_config.classes == "2-class": # type: ignore
    validation_configuration = FormatBinaryImageClassificationMetrics(cmap='BuGn', ROC_PR_line="darkorange")
    test_configuration = FormatBinaryImageClassificationMetrics(cmap='BuPu', ROC_PR_line="forestgreen")
else:
    validation_configuration = FormatMultiClassImageClassificationMetrics(cmap='YlGn', ROC_PR_line="darkorange")
    test_configuration = FormatMultiClassImageClassificationMetrics(cmap='Oranges', ROC_PR_line="forestgreen")

trainer.evaluate(save_dir=PM.metrics, 
                 model_checkpoint="best",
                 classification_threshold=0.5,
                 test_data=vision_dataset.test_dataset,
                 val_format_configuration=validation_configuration,
                 test_format_configuration=test_configuration
                 )
```

```python
# Finalizer
if train_config.classes == "2-class":  # type: ignore
    finalizer = FinalizeBinaryImageClassification(filename=train_config.finalized_file, # type: ignore
                                                classification_threshold=0.5,
                                                class_map=class_map)
else:
    finalizer = FinalizeMultiClassImageClassification(filename=train_config.finalized_file,  # type: ignore
                                                      class_map=class_map)

trainer.finalize_model_training(model_checkpoint="current",
                                save_dir=PM.artifacts,
                                finalize_config=finalizer)
```
