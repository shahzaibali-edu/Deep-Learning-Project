# CS-320 Deep Learning Semester Project — Progress & Workflow Log
**Project Title:** Rebuilding AlexNet in PyTorch (P01 — NeurIPS 2012)  
**Overarching Research Question:** *Which of AlexNet's ingredients really matter, and does training from scratch still compete with ImageNet pretraining?*  
**Institution:** National University of Sciences and Technology (NUST) — NBC, Quetta  
**Course & Instructor:** CS-320 Artificial Neural Networks & Deep Learning (Fall 2026) · Dr. Zirak Khan  
**GitHub Repository:** [https://github.com/shahzaibali-edu/Deep-Learning-Project](https://github.com/shahzaibali-edu/Deep-Learning-Project)

### Team Members & Role Split
| Team Member | CMS ID | Assigned Role | Core Module & Responsibilities |
| :--- | :--- | :--- | :--- |
| **Shahzaib Ali** | 528449 | Group Lead & Paper Lead | Architecture design, Convolutional stack (`Conv1`–`Conv5`), LRN, Overlapping MaxPool, weight initialization, and paper reproduction. |
| **Mubashara Anees** | 526720 | Data Lead | Custom `Dataset`/`DataLoader` pipelines, training-only normalization stats, data augmentations, tensor shape verification, and evaluation metrics. |
| **Abdul Ahad Khan** | 524914 | Experiments Lead | Fully-connected classifier head (`FC1`–`FC3`) with Dropout, sanity checks, mixed-precision training loop, ablations, Gradio demo, and report. |

---

## 1. Datasets & Where Each Is Used

To evaluate all claims of Krizhevsky et al. (2012) at Google Colab (T4 GPU) scale, the project uses three primary datasets (plus one optional bonus dataset):

1. **CIFAR-10 ($32 \times 32 \times 3$, 10 classes)**
   * **Links:** [Official CIFAR Page](https://www.cs.toronto.edu/~kriz/cifar.html) | [PyTorch CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
   * **Purpose (Stage 2 Reproduction Part A & RQ1):** Used **exclusively** with a small 4-layer CNN (not full 8-layer AlexNet) to reproduce **Figure 1** from Section 3.1 of the paper, testing whether ReLU reaches a $25\%$ training error rate approximately $6\times$ faster than $\tanh$ neurons without regularization.
2. **Imagenette ($224 \times 224 \times 3$, 10 classes, ~9,469 train / 3,925 val images)**
   * **Links:** [Fast.ai Imagenette GitHub](https://github.com/fastai/imagenette) | [PyTorch Imagenette](https://pytorch.org/vision/stable/generated/torchvision.datasets.Imagenette.html)
   * **Purpose (Stage 2 Reproduction Part B, M1–M2 Baseline & Core Stage 4 Ablations):** Serves as the Colab-scale replacement for full ImageNet (1.2M images). Used to train our rebuilt 8-layer AlexNet (`AlexNetRebuilt`), verify sanity checks, generate M2 baseline curves, compare Top-1/Top-5 error against Table 1 of the paper ($37.5\%$ Top-1, $17.0\%$ Top-5), and run Core ablation studies (RQ2).
3. **Oxford-IIIT Pets ($224 \times 224 \times 3$, 37 pet breeds, CC BY-SA 4.0)**
   * **Links:** [Official Oxford Pets Page](https://www.robots.ox.ac.uk/~vgg/data/pets/) | [PyTorch OxfordIIITPet](https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html)
   * **Purpose (Stage 3 Extension, Intermediate Stage 4 Ablations, Stage 5 Error Analysis & RQ3):** A fine-grained dataset the original paper never used. Used to compare a simple CNN baseline against 8-layer AlexNet trained **from scratch** vs. **ImageNet-pretrained AlexNet** (`torchvision.models.alexnet`), carry over the 5 ablations, analyze confusion matrices, and visualize the $96$ first-layer $11 \times 11 \times 3$ kernels (Figure 3).
4. **Fish-Pak (Optional Bonus Dataset — up to $+20\%$ grade)**
   * **Links:** [Mendeley Data (CC BY 4.0)](https://data.mendeley.com/datasets/n3ydw29sbz/3) | [Kaggle Mirror](https://www.kaggle.com/datasets/sitamul/fish-pak)
   * **Purpose:** Local Pakistani carp species classification dataset for bonus evaluation.

---

## 2. Completed Work So Far (Milestones M0, M1, and M2)

### Milestone M0: Environment, Fast Data Caching & Training-Split Normalization
* **Reproducibility & Hardware Setup:** Configured Google Colab T4 GPU runtime with deterministic random seeds (`seed = 42` across `random`, `numpy`, `torch`, and `cudnn`) and mounted Google Drive (`/content/drive/MyDrive/CS320_AlexNet_Checkpoints`) for persistent storage.
* **Fast `.tar.gz` Drive Caching Architecture:** Bypassed slow academic servers (`cs.toronto.edu` and `robots.ox.ac.uk`, which throttled downloads to $\sim 198\text{ kB/s}$) by using 16-connection `aria2c` and AWS S3 mirrors. Stored the 4 compressed archives (`cifar-10-python.tar.gz`, `imagenette2-320.tgz`, `images.tar.gz`, `annotations.tar.gz`) in `CS320_AlexNet_Checkpoints/tar_cache/`. Future sessions copy the compressed archives to local Colab SSD (`/content/data`) and unpack in $\sim 12\text{ seconds}$.
* **Leak-Free Normalization Statistics:** Computed channel-wise mean and standard deviation strictly on the **training splits only** and cached them in `dataset_stats.pt`:
  * **CIFAR-10 (Train):** $\text{Mean} = [0.4914, 0.4822, 0.4465]$, $\text{Std} = [0.2470, 0.2435, 0.2616]$
  * **Imagenette (Train):** $\text{Mean} = [0.4654, 0.4545, 0.4254]$, $\text{Std} = [0.2761, 0.2717, 0.2925]$
  * **Oxford-IIIT Pets (Trainval):** $\text{Mean} = [0.4829, 0.4448, 0.3957]$, $\text{Std} = [0.2592, 0.2533, 0.2598]$

### Milestone M1: Architecture Walkthrough, Hand Math & Custom Data Pipeline
* **Custom Dataset & DataLoader (Mubashara's Module):** Implemented `CustomVisionDataset` wrapping `torchvision` datasets with paper-faithful preprocessing (Section 2 & 4.1: resize shorter side to $256 \to$ random $224 \times 224$ patch extraction and horizontal reflection $p=0.5$ for training; $224 \times 224$ center crop for validation). Verified batch tensor shapes (`[128, 3, 224, 224]`, `torch.float32`) and plotted unnormalized sample grids with ground-truth labels.
* **Key Equations Documented:**
  * **ReLU Activation:** $f(x) = \max(0, x)$
  * **Local Response Normalization (LRN):**
    $$b_{x,y}^i = a_{x,y}^i \left/ \left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2\right)^\beta \right.$$
    using paper hyperparameters $k=2, n=5, \alpha=10^{-4}, \beta=0.75$.
* **Layer-by-Layer Tensor Shapes & Hand Parameter Count (Shahzaib & Abdul Ahad's Modules):**
  Verified in code via runtime `assert` statements and `torchinfo.summary()`:

| Layer | Configuration (Filters, Kernel, Stride, Padding) | Output Tensor Shape $(B, C, H, W)$ | Hand Parameter Calculation $(\text{Weights} + \text{Biases})$ | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Normalized RGB Image | $(B, 3, 224, 224)$ | $0$ | $0$ |
| **Conv1** | $96, 11\times11, s=4, p=2$ + ReLU | $(B, 96, 55, 55)$ | $(11 \times 11 \times 3 \times 96) + 96$ | $34,944$ |
| **LRN1 + Pool1** | LRN($n=5$) $\to$ MaxPool $3\times3, s=2$ | $(B, 96, 27, 27)$ | $0$ | $0$ |
| **Conv2** | $256, 5\times5, s=1, p=2$ + ReLU | $(B, 256, 27, 27)$ | $(5 \times 5 \times 96 \times 256) + 256$ | $614,656$ |
| **LRN2 + Pool2** | LRN($n=5$) $\to$ MaxPool $3\times3, s=2$ | $(B, 256, 13, 13)$ | $0$ | $0$ |
| **Conv3** | $384, 3\times3, s=1, p=1$ + ReLU | $(B, 384, 13, 13)$ | $(3 \times 3 \times 256 \times 384) + 384$ | $885,120$ |
| **Conv4** | $384, 3\times3, s=1, p=1$ + ReLU | $(B, 384, 13, 13)$ | $(3 \times 3 \times 384 \times 384) + 384$ | $1,327,488$ |
| **Conv5** | $256, 3\times3, s=1, p=1$ + ReLU | $(B, 256, 13, 13)$ | $(3 \times 3 \times 384 \times 256) + 256$ | $884,992$ |
| **Pool3** | MaxPool $3\times3, s=2$ | $(B, 256, 6, 6)$ | $0$ | $0$ |
| **Flatten** | Flatten spatial dimensions | $(B, 9216)$ | $0$ | $0$ |
| **FC1** | Dropout($0.5$) $\to$ Linear($9216 \to 4096$) + ReLU | $(B, 4096)$ | $(9216 \times 4096) + 4096$ | $37,752,832$ |
| **FC2** | Dropout($0.5$) $\to$ Linear($4096 \to 4096$) + ReLU | $(B, 4096)$ | $(4096 \times 4096) + 4096$ | $16,781,312$ |
| **FC3 (Head)** | Linear($4096 \to 10$) | $(B, 10)$ | $(4096 \times 10) + 10$ | $40,970$ |
| **Total** | **8 Learned Layers** | — | — | **58,322,314** |

*(Note: With $1000$ output classes as in ILSVRC, `FC3` has $(4096 \times 1000) + 1000 = 4,097,000$ parameters, bringing the total to $62,378,344 \approx 60\text{M}$ parameters.)*

### Milestone M2: Sanity Checks & 15-Epoch Baseline Run on Imagenette
* **Sanity Check 1 (Initial Loss $\approx \ln C$):** Verified initial untrained Cross-Entropy loss is $\approx 2.30\text{–}2.65$, close to theoretical uniform guessing $\ln(10) = 2.3026$.
* **Sanity Check 2 (Gradient Flow):** Confirmed non-zero mean absolute gradients across all 16 weight and bias tensors (`conv1.weight` through `fc3.bias`).
* **Sanity Check 3 (Single-Batch Overfit):** Verified that a 16-image batch with Dropout disabled ($p=0.0$), `kaiming_normal_(mode='fan_in')` initialization, and gradient clipping (`max_norm=1.0`) converges smoothly from $\approx 2.30$ to near-zero loss in 100 iterations.
* **15-Epoch Baseline Training (`m2_baseline.pt`):** Trained `AlexNetRebuilt` on Imagenette using SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=5e-4`, `batch_size=128`), `ReduceLROnPlateau(factor=0.1, patience=3)`, FP16 mixed precision (`torch.amp.autocast('cuda')`), and per-epoch Drive checkpointing:
  * **Epoch 01/15:** Train Loss: `2.3477` (`14.87%` Acc) | Val Loss: `2.1155` (`27.57%` Acc)
  * **Epoch 05/15:** Train Loss: `1.6785` (`42.67%` Acc) | Val Loss: `1.5280` (`49.63%` Acc)
  * **Epoch 10/15:** Train Loss: `1.3850` (`53.97%` Acc) | Val Loss: `1.3341` (`55.75%` Acc)
  * **Epoch 15/15:** Train Loss: `1.1945` (`61.20%` Acc) | Val Loss: `1.1380` (`63.72%` Acc)

---

## 3. Bug Diary (Completed Report Appendix Requirement — 3/3 Bugs Logged)

The project rubric requires documenting at least three real bugs encountered during development in the format **Symptom $\to$ Cause $\to$ Fix**.

### Bug #1: Single-Batch Overfit Sanity Check Plateauing at High Loss
* **Symptom:** During Sanity Check 3 (overfitting 16 images for 100 steps), the training loss started at `2.3230` ($\approx \ln 10$) but plateaued at `2.1005` instead of dropping to near-zero.
* **Cause:** Inspecting per-layer gradients in Sanity Check 2 revealed a $1000\times$ gradient magnitude imbalance between early convolutional layers (`conv2.weight` mean abs grad $= 4.03 \times 10^{-5}$) and the final classification head (`fc3.weight` mean abs grad $= 1.09 \times 10^{-1}$). The original 2012 paper's manual Gaussian initialization ($\mathcal{N}(0, 0.01^2)$ with hidden biases $= 1.0$) caused severe signal attenuation across 8 unnormalized layers during early steps.
* **Fix:** Replaced fixed $\mathcal{N}(0, 0.01^2)$ initialization in hidden layers with Kaiming/He normal initialization (`nn.init.kaiming_normal_`), which scales variance by $\sqrt{2/\text{fan\_in}}$ to preserve activation variance across ReLU layers, and set initial biases to $0.0$.

### Bug #2: Loss Explosion to `NaN` ($6 \times 10^{12}$) Under `fan_out` Kaiming Initialization
* **Symptom:** After switching `AlexNetRebuilt` to `nn.init.kaiming_normal_(mode='fan_out')`, Step 1 loss jumped to `7.3483` and exploded to $6.3 \times 10^{12}$ (`nan`) by iteration 3.
* **Cause:** In the final linear layer `fc3` ($4096 \to 10$), using `mode='fan_out'` scaled weights by $\sqrt{2/10}$ instead of $\sqrt{2/4096}$, making initial output weights $\sim 20\times$ too large. Under `SGD(lr=0.01, momentum=0.9)` without Batch Normalization, these large initial logits triggered an exploding gradient cascade.
* **Fix:** Changed Kaiming initialization to `mode='fan_in'` for all Conv/Linear layers, kept `fc3` weights initialized to $\mathcal{N}(0, 0.01^2)$ so initial logits start near zero ($\text{Loss} \approx 2.30$), lowered base LR to `0.005`, and added unscaled gradient norm clipping (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`).

### Bug #3: Stale Google Drive Checkpoint Reloading & Freezing Training at 10.49% Accuracy
* **Symptom:** When running the 15-epoch Imagenette training loop after fixing weight initialization, the notebook printed `Resumed from epoch 12` and remained frozen at `Train Loss: 2.3019 | Train Acc: 10.49% | Val Acc: 9.10%` through Epoch 15.
* **Cause:** The automatic checkpoint resume logic detected `m2_baseline.pt` saved on Google Drive from an earlier failed run (which had used the unfixed weights and where `ReduceLROnPlateau` had already decayed the learning rate to $10^{-5}$), overwriting our newly initialized Kaiming weights with stalled weights and a dead learning rate.
* **Fix:** Added an explicit `reset_checkpoint` parameter to `train_model()` and purged the stale `m2_baseline.pt` file before starting a fresh run, resulting in immediate convergence from `14.87%` (Epoch 1) to `63.72%` validation accuracy (Epoch 15).

---

## 4. Critical Precautions & Pitfalls to Avoid

1. **Preventing Data Leakage:**
   * **Normalization Stats:** Never compute mean or standard deviation on validation or test splits. Always use the training split statistics stored in `dataset_stats.pt` to normalize both `train_loader` and `val_loader`.
   * **Augmentation:** Random cropping (`RandomCrop(224)`), random horizontal flipping, and PCA RGB color jittering must **only** be applied to `train_loader`. Validation and test loaders must strictly use deterministic `Resize(256)` followed by `CenterCrop(224)`.
2. **Train vs. Eval Mode (`model.train()` vs. `model.eval()`):**
   * AlexNet uses two heavy `Dropout(p=0.5)` layers in `FC1` and `FC2`. Forgetting `model.eval()` during validation/testing leaves Dropout active and corrupts accuracy metrics. Conversely, leaving `model.eval()` on during training disables Dropout and causes massive overfitting.
   * *Viva Tip:* Be ready to explain why validation accuracy (`63.72%`) is slightly higher than training accuracy (`61.20%`) at Epoch 15—during training, `Dropout(0.5)` zeroes out half of the hidden classifier activations and random cropping makes inputs harder, whereas validation uses all neurons with scaled weights and clean center crops.
3. **Strict One-Factor-at-a-Time (OFAT) Control in Ablations:**
   * In Milestone M4, all 5 ablation experiments must start from the **exact same random seed (`set_seed(42)`)**, the **exact same train/val split**, and the **exact same number of epochs and optimizer settings**, altering only one hyperparameter/component per run and saving each to its own unique checkpoint file (e.g., `ablation_no_dropout.pt`, `ablation_no_lrn.pt`).
4. **Google Drive & GitHub Hygiene:**
   * **Never hardcode checkpoint deletion:** Keep `reset_checkpoint=False` in `train_model()` so re-running cells after a Colab disconnect resumes cleanly or loads completed weights without wiping progress.
   * **Google Drive Sharing:** Ensure Mubashara and Abdul Ahad add a Drive shortcut to `CS320_AlexNet_Checkpoints` directly inside their `My Drive` root so `/content/drive/MyDrive/CS320_AlexNet_Checkpoints` resolves identically for all three team members.
   * **Individual GitHub Commits:** The instructor verifies GitHub commits from all three members individually. Never commit `/content/data` or `.pt` checkpoint files to GitHub; each member must commit their respective code sections and ablations from their own GitHub account.

---

## 5. Future Work & Step-by-Step Roadmap (Milestones M3 to M5)

### Milestone M3: Paper Reproduction (Due Week 11 Lab — November 20, 2026)
* **Task 3A — Reproduce Figure 1 on CIFAR-10 (RQ1):**
  * Build a 4-layer CNN for $32 \times 32$ CIFAR-10 images with zero regularization (no dropout, no weight decay, no augmentation).
  * Train two versions with independently tuned learning rates: one using **ReLU** and one using **$\tanh$**.
  * Plot **Training Error Rate vs. Epochs** ($0$ to $40$ epochs, y-axis $0.0$ to $0.75$) beside the paper's Figure 1 (`image_f0e502.png`), marking the exact epoch each network crosses the $25\%$ ($0.25$) training error threshold to verify the $\sim 6\times$ speedup.
* **Task 3B — Imagenette Top-1 & Top-5 Comparison Table:**
  * Extend `m2_baseline.pt` training on Imagenette (e.g., to 25–35 epochs) and compute **Top-1 Error Rate** and **Top-5 Error Rate** (both single-crop and 10-crop test averaging as described in Section 4.1).
  * Present a side-by-side table comparing your Imagenette numbers with the paper's ILSVRC-2010 results ($37.5\%$ Top-1 error, $17.0\%$ Top-5 error) and explain the gap (10 classes vs. 1,000 classes; $\sim 9.5\text{k}$ vs. $1.2\text{M}$ images; single T4 GPU vs. 2-GPU column split).

### Milestone M4: New Dataset Extension & Ablation Study (Due Week 13 Lab — December 4, 2026)
* **Task 4A — Oxford-IIIT Pets Extension (RQ3):**
  * Split `OxfordIIITPet` (`trainval`) into a fixed training and validation split (`set_seed(42)`).
  * Train three models on the 37 pet breeds:
    1. **Simple Baseline:** A lightweight 3–4 layer CNN.
    2. **AlexNet from Scratch:** `AlexNetRebuilt(num_classes=37)` initialized with Kaiming weights.
    3. **AlexNet Pretrained:** `torchvision.models.alexnet(weights='DEFAULT')` with the final `classifier[6]` replaced by `nn.Linear(4096, 37)`.
* **Task 4B — Five One-Factor-at-a-Time Ablations (RQ2):**
  * Run 5 controlled experiments from the fixed baseline configuration (each member runs and commits at least one):
    1. **Dropout:** On ($p=0.5$) vs. Off ($p=0.0$).
    2. **Pooling:** Overlapping MaxPool ($z=3, s=2$) vs. Non-overlapping MaxPool ($z=2, s=2$).
    3. **Normalization:** Local Response Normalization (LRN) On vs. Off.
    4. **Data Augmentation:** Random crop + flip + PCA RGB color jitter On vs. Off (center crop only).
    5. **Weight Decay:** $5 \times 10^{-4}$ vs. $0.0$.
  * Compile all results into one master table reporting `Train Acc`, `Val Acc`, `Train-Val Gap`, and `Delta from Baseline`.
* **Task 4C — Error Analysis & First-Layer Kernel Visualization (Figure 3):**
  * Generate confusion matrices and plot top-5 probability bars on misclassified test images (mirroring Figure 4 of the paper).
  * Extract the $96$ `conv1` filters ($11 \times 11 \times 3$) from both the scratch and ImageNet-pretrained models to compare learned edge/frequency/color blob detectors against Figure 3 of the paper.

### Milestone M5 & Final Deliverables (Weeks 15–17 — December 18 to January 1)
* **Architecture One-Pager (Week 15):** 1-page printed summary for the class handbook featuring the tensor-shape diagram, key equations, hyperparameters, model strengths/weaknesses, and 3 self-check questions.
* **Gradio Live Demo (Week 16):** Interactive web interface with an image upload panel, top-5 class prediction probabilities, and a live visualization of internal `conv1` activations/kernels.
* **Final 5–6 Page Report & Individual Viva (Weeks 16–17):** Complete the structured report (Paper 1p, Reproduction 1p, Pets Dataset 1p, Ablations & Debugging 1.5p, Error Analysis 0.5p, Limitations/Contributions/AI-use 0.5p + Appendix with Bug Diary and References) and prepare for live hyperparameter modifications during the individual viva.