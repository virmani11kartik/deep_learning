# MNIST Representation Learning — From Scratch and With PyTorch 🧠📊

This project is a **self-learning initiative** to explore how classical and modern machine learning methods perform on the MNIST digit classification task.  
I implemented everything from scratch using **NumPy** and compared it against **PyTorch** implementations, analyzing results through training/validation curves and feature visualizations.

---

## 🔍 Motivation
Instead of relying only on high-level frameworks, I wanted to understand:
- How a neural network actually performs **forward and backward passes** at the layer level.
- How **gradient checking** validates correctness of custom backprop implementations.
- The effect of different **feature representations** (raw pixels vs. Gabor filters).
- How a scratch-built pipeline compares to **PyTorch** in terms of stability and accuracy.

---



---

## 🛠️ Implementations

### 1. Feature Engineering + SVM
- Downsampled MNIST digits to $14\times14$ resolution.
- Built **Gabor filter banks** to extract orientation- and frequency-sensitive features.
- Trained **SVM classifiers** on:
  - Raw pixel inputs
  - Gabor-enhanced features
- Result: Gabor features improved accuracy and yielded more separable confusion matrices.

### 2. Neural Networks From Scratch (NumPy)
- Built custom layers:
  - **Embedding**: 28×28 → 7×7×8 patches
  - **Linear**: fully connected mapping
  - **ReLU** activation
  - **Softmax + Cross-Entropy** loss
- Verified each component with **gradient checking** (finite differences).
- Trained using minibatch SGD (batch=32, lr=0.1) for 10k updates.
- Observed ~40–60% accuracy with simple 1-layer architecture.

### 3. Neural Networks With PyTorch
- Reimplemented the same pipeline using `nn.Conv2d`, `nn.Linear`, `nn.ReLU`, and `nn.CrossEntropyLoss`.
- Trained for 10k updates with identical settings.
- Achieved smoother convergence, better stability, and higher final accuracy compared to NumPy model.

---

## 📊 Results

### SVM Experiments
- **Raw pixels**: struggled with invariances (rotation/scale).
- **Gabor features**: added robustness to orientation and frequency, improving classification.

### Neural Networks
- **NumPy model**:  
  - Showed correct learning behavior but limited by shallow architecture.  
  - Training error stabilized at ~0.4–0.6.
- **PyTorch model**:  
  - More stable training curves.  
  - Better validation error.  
  - Demonstrated advantages of optimized layers.

Results are visualized in the `outputs_p3i/` and `Support_Vector_Machine/` folders:

- Training loss/error curves (NumPy vs PyTorch)
- Validation loss/error curves
- Gabor filter visualizations
- Confusion matrices

---

## 📊 Results Preview

### Neural Network (NumPy vs PyTorch)

**Validation Loss (NumPy)**  
![Validation Loss](Basic_Neural_Net/val_loss_vs_updates.png)

**Validation Error (NumPy)**  
![Validation Error](Basic_Neural_Net/val_error_vs_updates.png)

**PyTorch Training Loss**  
![PyTorch Train Loss](Basic_Neural_Net/outputs_p3i/train_loss.png)

**PyTorch Validation Error**  
![PyTorch Val Error](Basic_Neural_Net/outputs_p3i/val_error.png)

---

### SVM with Gabor Filters

**Filter Bank (12 Gabor filters)**  
![Gabor Filter Bank](Support_Vector_Machine/outputs_part_h/gabor_filters_visualization.png)

**Confusion Matrix: Baseline vs Gabor Features**  
![Gabor Filter Bank](Support_Vector_Machine/outputs_part_h/gabor_vs_baseline_confusion.png)

# ResNet / All-CNN Experiments (CIFAR-10)

This repository contains:

* `resnet.py` — (uploaded) TorchVision-style ResNet implementation (ResNet-18/34/50/...)
* `param_count.py` — layer-wise parameter counting / simple diagnostics
* `allcnn.py` — training script for the All-CNN-T style model on CIFAR-10
* `grad_and_attack_analysis.py` — compute input gradients and run iterative FGSM attacks
* checkpoint outputs in `checkpoints/` and analysis outputs in `adv_analysis/` (generated during runs)
* utility scripts / notebooks used during the assignment

---

## Project overview

This repo explores:

1. Convolutional model design (All-CNN-T), and ResNet implementation nuances

   * We use BN → ReLU ordering (ResNet v1 / v1.5). This is the canonical pattern: BatchNorm normalizes the pre-activation distribution, then ReLU applies the nonlinearity. Pre-activation ResNets (BN→ReLU→Conv) are an alternate design for very deep networks.
2. Model inspection and parameter bookkeeping:

   * Count params per-layer, and split parameters into groups for weight decay handling (no decay on biases and BN affine params).
3. Training on CIFAR-10 with augmentation and a standard SGD+Nesterov schedule.
4. Gradient-based adversarial attacks (FGSM / iterative FGSM) and empirical evaluation of robustness.

---

## Quick setup

Create a Python virtualenv and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision tqdm matplotlib numpy
```

> If you have a CUDA GPU, install the appropriate `torch`/`torchvision` wheel for your CUDA version from [https://pytorch.org](https://pytorch.org).

---

## Files & purpose

* `resnet.py` — ResNet family implementation (for inspection / experiments).
* `param_count.py` — script to print per-parameter tensor counts and totals.
* `allcnn.py` — main CIFAR-10 training script (All-CNN-T). Produces:

  * `checkpoints/allcnn_t_best.pth` (best val)
  * `checkpoints/allcnn_t_last.pth` (final)
  * `training_curves.png` and `training_curves.json`
* `grad_and_attack_analysis.py` — computes `∂L/∂x` visualizations and runs the 5-step signed-gradient attack; saves images and `attack_loss_curve.png`.
* `adv_analysis/` — directory where gradient visualizations and attack plots are saved by the analysis script.

---

## Model notes (All-CNN-T)

The architecture implemented in `allcnn.py` exactly matches the All-CNN-T pattern used in the assignment:

* Small conv blocks, ReLU, BatchNorm, Dropout
* Global avg pooling to produce logits
* Example constructor: `allcnn_t(c1=96, c2=192)`

Important design choices:

* Convolutions are created with `bias=True` in that script (it’s fine either way when not relying on BN affine), but when using BN, conv biases are usually redundant (common practice: `bias=False`).
* BatchNorm is placed before ReLU: `Conv → BN → ReLU`.

---

## Training instructions (CIFAR-10)

Train with recommended hyperparams:

```bash
python allcnn.py --data ./data --batch-size 128 --epochs 100 --amp
```

Key training settings in the script:

* Augmentations: `RandomHorizontalFlip`, `RandomCrop(32, padding=4, padding_mode="reflect")`, `ColorJitter(brightness=0.2, contrast=0.2)` plus normalization.
* Optimizer: `SGD` with Nesterov momentum `0.9`.
* Weight decay: `1e-3` applied **only** to weight parameters (not to biases or BN affine params). The script implements param grouping: decay group vs no-decay group.
* LR schedule: piecewise step schedule:

  * `lr=0.1` for epochs 0–39,
  * `lr=0.01` for epochs 40–79,
  * `lr=0.001` for epochs 80–99.
* Dropout & BatchNorm included in model.
* Mixed-precision optional: `--amp`.

Expected outcome:

* With the default channels `c1=96, c2=192`, and the augmentations + schedule above, validation error below ~10% is achievable on CIFAR-10; if you fall short, try:

  * increasing `c1/c2` to `128`/`256`,
  * adding Cutout,
  * longer training or cosine LR schedule.

---

## Parameter counting & grouping

**Layer-wise parameter count** (example command, or open and run `param_count.py`):

```bash
python param_count.py
```

Example output for ResNet-18 (canonical):

* Total trainable params: `11,689,512`

**Parameter grouping** (used by `allcnn.py`):

* `bn_affine_params`: BatchNorm `weight` and `bias` (γ and β) → *no weight decay*
* `bias_params`: biases of `Conv2d` and `Linear` → *no weight decay*
* `rest_params`: all other parameters → *apply weight decay = 1e-3*

This is implemented in `get_param_groups(model)` in the training script.

---

## Data augmentation

RandAugment and many modern strategies are useful; for assignment requirements we used:

* `RandomHorizontalFlip`
* `RandomCrop` with `padding=4` (reflect)
* `ColorJitter` (brightness, contrast)
* (Optional) Cutout — helpful if accuracy is slightly below target.

If you want to mirror the RandAugment paper/implementation, inspect torchvision transforms:

* `torchvision.transforms.RandAugment` (see torchvision docs).

---

## Gradient & adversarial analysis

We compute input gradients `dx = ∂L/∂x` (backprop into the input) and visualize them and run small iterative attacks.

Key points / how it was implemented:

* **Compute input gradient**:

  ```py
  x = x_batch[i:i+1].detach().clone().requires_grad_()
  logits = net(x)
  loss = F.cross_entropy(logits, y)
  loss.backward()
  dx = x.grad.data.clone()
  ```
* **5-step signed-gradient attack (iterative FGSM)**:

  * Total `ε = 8` in pixel units `[0..255]`.
  * Step size `α = ε / 5`.
  * Convert pixel-space step to normalized-space before applying to the model:

    * `eps_norm[c] = (eps_pixels / 255) / std[c]`
    * `alpha_norm[c] = (alpha_pixels / 255) / std[c]`
  * At each step:

    * compute gradient wrt normalized input,
    * `x = clamp(x + alpha_norm * sign(grad), orig - eps_norm, orig + eps_norm)`
* **Outputs**:

  * `adv_analysis/grad_visuals_correct.png`
  * `adv_analysis/grad_visuals_incorrect.png` (if misclassified examples exist)
  * `adv_analysis/attack_loss_curve.png` (mean batch loss vs attack step)
  * `adv_analysis/attack_image_progression_example0.png` (example progression)

**Compute adversarial accuracy (1-step FGSM)**:

* To compute accuracy on 1-step perturbed images (for the entire validation set), generate for each validation image:

  * `x_adv = x + eps * sign(∂L/∂x)` (careful to convert `eps` from pixel units into normalized-space before applying),
  * clamp `x_adv` into valid normalized bounds (or equivalently clamp the pixel values to `[0,255]` then re-normalize),
  * feed `x_adv` to the model and check predictions.
* Typical observation: accuracy on 1-step FGSM images is substantially lower than on clean images, demonstrating network fragility. (~significant drop depending on epsilon and model).

A quick snippet to do this over the validation set (pseudocode):

```py
eps_pixels = 8.0
eps_norm = torch.tensor((eps_pixels/255.0)/std).view(1,3,1,1).to(device)

correct = 0; total = 0
for x, y in val_loader:
    x = x.to(device); y = y.to(device)
    x_adv = x.detach().clone().requires_grad_()
    logits = net(x_adv); loss = F.cross_entropy(logits, y); loss.backward()
    grad = x_adv.grad
    x_adv = x + eps_norm * grad.sign()
    # project: clamp to orig +/- eps_norm OR clamp pixel range and re-normalize
    logits_adv = net(x_adv)
    pred_adv = logits_adv.argmax(dim=1)
    correct += (pred_adv == y).sum().item()
    total += y.size(0)
adv_acc = correct / total
```

**Compare** `adv_acc` to the clean validation accuracy to quantify robustness.

---

## Visuals (generated during analysis)

Below are the key images we generated during the experiments. If you ran the training & analysis scripts, these files should exist in the repository at the shown paths.

### Training curves

![Training Curves](training_curves.png)
*Figure: Training and validation loss (left) and top-1 error (right) vs epochs.*

### Adversarial attack loss curve

![Attack Loss Curve](adv_analysis/attack_loss_curve.png)
*Figure: Mean cross-entropy loss on a mini-batch vs the 5 signed-gradient attack steps (total ε=8 pixels).*n

### Gradient visualizations

Correctly classified examples (top-row images) and their input gradients (bottom-row):

![Grad visuals - correct](adv_analysis/grad_visuals_correct.png)

Misclassified examples (if present) and their gradients:

![Grad visuals - incorrect](adv_analysis/grad_visuals_incorrect.png)

### Example perturbed image progression

![Perturbation progression example](adv_analysis/attack_image_progression_example0.png)
*Figure: how a single image changes across the 5 signed-gradient attack steps (visualized in pixel space).*

> If any of the images above are missing, run:
>
> ```bash
> python allcnn.py --data ./data --batch-size 128 --epochs 100 --amp
> python grad_and_attack_analysis.py
> ```

---

## Troubleshooting & common gotchas

* **Interleaved tqdm + print lines**
  If you use `tqdm` for progress and also `print(...)`, logs can interleave. Use one of:

  * `tqdm.write(msg)` instead of `print(...)`, or
  * `pbar.set_postfix(...)` to show train/val metrics inline, or
  * disable the progress bar.

* **NumPy and PyTorch mixing error (`TypeError: unsupported operand...`)**
  When multiplying a NumPy array by a PyTorch tensor, convert the NumPy array to a tensor first (and move to the same device). Example:

  ```py
  alpha_norm = torch.tensor(alpha_norm_np, device=device).view(1,3,1,1)
  x_adv = x_adv + alpha_norm * grad.sign()
  ```

* **Normalization must match training**
  Always use the exact `mean` and `std` used for training when converting `eps_pixels` to normalized space. Otherwise perturbation magnitudes are incorrect.

* **Epoch counts & schedule**
  If you run `--epochs 200`, adjust the LR schedule to 80/80/40 (or change `get_lr()` accordingly). If you want the assignment schedule for 100 epochs, pass `--epochs 100`.

* **Gradients are zero**
  Ensure input tensor has `requires_grad_()` before forward, and that you call `loss.backward()`.

---

## Reproducibility tips

* Set `seed` in `allcnn.py` (example uses `42`). Note that exact reproducibility on GPU requires disabling cudnn benchmark and setting deterministic flags (but may run slower).
* Use `--amp` if you want mixed precision when using modern GPUs. Remove it if you want exact gradients in float32.

---

## Expected outputs (files)

After training and analysis, you should have:

* `checkpoints/allcnn_t_best.pth`
* `checkpoints/allcnn_t_last.pth`
* `training_curves.png` and `training_curves.json`
* `adv_analysis/grad_visuals_correct.png`
* `adv_analysis/grad_visuals_incorrect.png`
* `adv_analysis/attack_loss_curve.png`
* `adv_analysis/attack_image_progression_example0.png`

---

## References

* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition (ResNet).
* Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples (FGSM).
* Cubuk, E. D., et al. (2019). RandAugment: Practical automated data augmentation with a reduced search space.
* Torchvision transform docs: `torchvision.transforms` (RandAugment implementation referenced during augment experiments).

---

## ⚙️ Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U numpy matplotlib torchvision torch tqdm scikit-learn
