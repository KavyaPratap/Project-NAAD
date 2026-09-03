# KWS Test Results

## Dataset
- Total keyword samples: [X] (before aug)
- Total negative samples: [X] (before aug)
- After augmentation (train only): [X] total
- Speaker-disjoint: YES (spk##-## in train, spk## in val, spk## in test)

## Float Model
- Architecture: DS-CNN (64 filters, 4 DS blocks)
- Parameters: [X]
- Training epochs: [X] (early stopped at [X])
- Val accuracy: [X]%
- Test recall: [X]%
- Test precision: [X]%
- Test FAR (test set FP count): [X]

## INT8 Model
- Model size: [X] KB
- Test accuracy vs float: delta = [X]% (must be < 2%)
- INT8 test recall: [X]%

## On-Device TFLM (ESP32-S3)
- Arena used: [X] KB / 96 KB
- Inference time: [X] us
- Score at trigger threshold (0.85): [X]%
- False activations in 60min negative test: [X]
- FAR = [X] / hour
