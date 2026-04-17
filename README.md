# detr_resnet_50_character_recognition
This repo is part of the class Selected Topics in Visual Recognition using Deep Learning of the NYCU.

## Experimental Setup & Testing Strategy

To achieve the highest possible Mean Average Precision (mAP) for digit detection, we employ a structured ablation approach. Each test below isolates a specific hyperparameter or architectural decision. These tests form the basis of our final model selection and can be directly cited in the project report.

### 1. The Baseline Configuration
**Command:**
```bash
python src/train.py --epochs 10 --batch-size 16 --lr 2e-4 --lr-backbone 2e-5
```
**Rationale:** 
Establishes a solid control group using standard Deformable DETR defaults (Cosine learning rate scheduler, 2e-5 backbone LR, 2e-4 transformer LR, 10 epochs). We reduced epochs and increased batch size to prioritize computational efficiency, taking advantage of Deformable DETR's 10x faster convergence compared to Vanilla DETR. We need this baseline to measure whether our subsequent architectural tweaks actually improve the model.

### 2. Extended Convergence & Heavy Regularization
**Command:**
```bash
python src/train.py --epochs 20 --batch-size 16 --weight-decay 2e-4
```
**Rationale:** 
Although Deformable DETR matches quickly, the Hungarian Matcher bounding box alignments might still take a bit longer to perfectly stabilize on small digits. By extending the epochs to 20, we allow full convergence while maintaining strict compute limits. To prevent the model from memorizing the training images over this extended duration, we double the `weight-decay` (from `1e-4` to `2e-4`).

### 3. Backbone Preservation (Extreme Differential LR)
**Command:**
```bash
python src/train.py --epochs 10 --batch-size 16 --lr 2e-4 --lr-backbone 5e-6
```
**Rationale:** 
The ResNet-50 weights are already heavily pre-trained. The new Deformable Transformer layers, however, are initialized from scratch. Using the baseline `2e-5` LR on the backbone might still be too aggressive and destroy the fine-grained edge-detection features ResNet already learned. Dropping the backbone LR to a conservative `5e-6` acts as a shield, freezing the ResNet's structural integrity while the decoder learns spatial mappings.

### 4. The Optimized Setup
**Command:**
```bash
python src/train.py --epochs 20 --batch-size 16 --weight-decay 2e-4 --lr-backbone 5e-6
```
**Rationale:** 
Combines the theoretical best parameters derived from the hypotheses above: extend the training duration to let the Hungarian Matcher perfectly settle, apply balanced regularization across the longer timeframe, and rigidly protect the pre-trained backbone.

*(Note: We do not need to test aggressive image upscaling parameters like `min-size=1000` because Deformable DETR natively solves the "small object problem" via its multi-scale cross-attention feature pyramid over ResNet's C3-C5 layers. Upscaling would only cause massive Out-Of-Memory errors and slowdowns without theoretical benefit).*

## Inference / Prediction

Once you have trained a model and selected the best weights, you can generate the required `pred.json` in COCO format for the test dataset using the `predict.py` script.

**Command:**
```bash
python src/predict.py --weights models/<TARGET_RUN_DIR>/best_map_model.pt --data datasets/test
```

**Notes:**
- Replace `<TARGET_RUN_DIR>` with the specific run folder generated in `models/` (e.g., `run_20260417_172357`).
- The script will process the images in `datasets/test` and automatically save a `pred.json` file inside the same directory as the weights file. This file is formatted exactly as required for the final evaluation submission.
