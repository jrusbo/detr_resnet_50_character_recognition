# detr_resnet_50_character_recognition
This repo is part of the class Selected Topics in Visual Recognition using Deep Learning of the NYCU.

## Experimental Setup & Testing Strategy

To achieve the highest possible Mean Average Precision (mAP) for digit detection, we employ a structured ablation approach. Each test below isolates a specific hyperparameter or architectural decision. These tests form the basis of our final model selection and can be directly cited in the project report.

### 1. The Baseline Configuration
**Command:**
```bash
python src/train.py --epochs 10 --batch-size 16 --lr 1e-4 --lr-backbone 1e-5 --min-size 800 --max-size 1333
```
**Rationale:** 
Establishes a solid control group using standard DETR defaults (Cosine learning rate scheduler, 1e-5 backbone LR, 10 epochs). We reduced epochs and increased batch size to prioritize computational efficiency. We need this baseline to measure whether our subsequent architectural "hacks" actually improve the model or just introduce instability.

### 2. Attacking the "Small Object" Problem (Upscaling)
**Command:**
```bash
python src/train.py --epochs 10 --batch-size 8 --min-size 1000 --max-size 1500
```
**Rationale:** 
DETR notoriously struggles with small objects because the ResNet-50 backbone downsamples the image features by a factor of 32. Tiny street-view digits can literally vanish between pixels. By intentionally upscaling the shortest edge to 1000 pixels (and capping the max edge at 1500), we force the numbers to occupy a larger spatial footprint in the transformer feature maps. *(Note: Batch size is reduced to 4 to prevent Out-Of-Memory errors on Kaggle T4 GPUs due to the larger tensors).*

### 3. Extended Convergence & Heavy Regularization
**Command:**
```bash
python src/train.py --epochs 20 --batch-size 16 --weight-decay 2e-4
```
**Rationale:** 
The Hungarian Matcher inside DETR takes significantly longer to stabilize compared to traditional CNN anchors like YOLO. By extending the epochs to 20, we allow full convergence while maintaining our strict compute limits. However, to prevent the model from memorizing the training images over this extended duration, we double the `weight-decay` (from `1e-4` to `2e-4`).

### 4. Backbone Preservation (Extreme Differential LR)
**Command:**
```bash
python src/train.py --epochs 10 --batch-size 16 --lr 1e-4 --lr-backbone 5e-6
```
**Rationale:** 
The ResNet-50 weights are already heavily pre-trained and highly capable of detecting low-level lines and curves (perfect for digits). The new Transformer layers, however, are initialized from scratch. If the backbone learning rate is too high, the chaotic early-epoch gradients from the untrained transformer will destroy the backbone's useful features. Dropping the backbone LR to a conservative `5e-6` acts as a shield, freezing the ResNet's structural integrity while the decoder learns to draw bounding boxes.

### 5. The "Kitchen Sink" (Combined Optimal Setup)
**Command:**
```bash
python src/train.py --epochs 20 --batch-size 8 --min-size 1000 --max-size 1333 --weight-decay 1.5e-4 --lr-backbone 5e-6
```
**Rationale:** 
Combines the theoretical best parameters derived from the hypotheses above: upscale the digits so they survive the RESNET bottleneck, extend the training duration to let the Hungarian Matcher settle, apply balanced regularization across the longer timeframe, and rigidly protect the pre-trained backbone.

## Inference / Prediction

Once you have trained a model and selected the best weights, you can generate the required `pred.json` in COCO format for the test dataset using the `predict.py` script.

**Command:**
```bash
python src/predict.py --weights models/<TARGET_RUN_DIR>/best_map_model.pt --data datasets/test
```

**Notes:**
- Replace `<TARGET_RUN_DIR>` with the specific run folder generated in `models/` (e.g., `run_20260417_172357`).
- The script will process the images in `datasets/test` and automatically save a `pred.json` file inside the same directory as the weights file. This file is formatted exactly as required for the final evaluation submission.
