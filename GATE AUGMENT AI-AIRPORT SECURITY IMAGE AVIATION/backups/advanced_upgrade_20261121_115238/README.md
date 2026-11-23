# GateAugment AI

GateAugment AI is a local Streamlit research application for baggage X-ray image augmentation and CNN robustness experiments. It supports persistent image upload, traditional image transformations, optional YOLO boxes, a pretrained Diffusers image-to-image pipeline, TensorFlow/Keras classifier training, held-out test evaluation, visual comparisons, experiment history, and CSV/JSON/image downloads.

> **Research-only:** This application is not an operational airport screening system. Review every generated image and its annotations. A generic pretrained diffusion model is not X-ray-domain validated and can change security-relevant content.

## Requirements and installation

- Python 3.10–3.12 recommended.
- A CUDA-capable GPU is optional for diffusion and TensorFlow acceleration.
- Diffusion requires model weights and may download a large model the first time.

Create and activate a virtual environment, then install:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation availability depends on the operating system and Python version. If TensorFlow cannot be installed in the current environment, image upload and augmentation remain usable, while classifier training reports that TensorFlow is unavailable. For GPU-specific PyTorch installation, follow the official PyTorch selector, then install optional diffusion packages:

```powershell
pip install diffusers transformers accelerate
```

## Run

```powershell
streamlit run app.py
```

## Workflow

1. **Image upload:** upload JPG, JPEG, or PNG images (maximum 25 MB per image). Upload metadata and decoded images are held in Streamlit session state.
2. **Image augmentation:** select rotation, flips, brightness/contrast, scaling, translation, crop, occlusion, or overlap composition. Overlap requires a second image with identical dimensions.
3. **Generative AI studio:** the application attempts a real Diffusers `StableDiffusionImg2ImgPipeline` run for the configured model ID. Model downloads may require internet access, disk space, and an accepted model license. The default model is generic, not trained or validated on X-ray imagery. If it is unavailable, the optional fallback is an Albumentations transformation and is explicitly labeled as traditional, not generative.
4. **Model training:** select a dataset under `dataset/`, then train a baseline CNN and a second CNN with Keras random flip/rotation/contrast layers applied to training samples only. The original train directory is split into training and validation subsets before fitting; the separate test directory is not used for fitting or augmentation. Both models are evaluated on the same test set. Epoch progress, model checkpoints, training histories, and experiment summaries are saved.
5. **Evaluation, comparison, and reports:** view actual accuracy, macro precision/recall/F1, per-class classification reports, confusion matrices, and training/validation curves. Compare images and actual model results. Export CSV/JSON reports and a ZIP of generated PNGs (and YOLO sidecars when supplied).

## Persistent image upload and annotations

Uploaded images are validated and saved under `assets/
  uploads/
  annotations/`. The project stores metadata, generated-image provenance, recent activity, and experiment summaries in `reports/project_state.json`; images and model files remain in their corresponding local folders. Upload one or more matching YOLO `.txt` files with the images to preserve boxes through geometric transformations. Each annotation file must share its image's filename stem and contain rows in this format:

```text
class_id x_center y_center width height
```

Coordinates and dimensions are normalized to `[0, 1]`. Augmented annotated images receive a matching `.txt` sidecar and are included in downloads. Overlap composition for annotated images requires annotations for both source images. The CNN in this project performs image-level classification; YOLO box support is for annotation-preserving image augmentation, not object-detection model training.

## Dataset setup

Datasets are **not bundled or downloaded automatically**. Obtain data from the dataset owners and comply with each dataset's license, research terms, and usage restrictions. Public research datasets to investigate include SIXray, OPIXray, and PIDray; availability and terms may change.

Arrange class-labeled classification images as:

```text
dataset/
  train/
    class_a/
      image_001.png
    class_b/
      image_002.png
  test/
    class_a/
      image_101.png
    class_b/
      image_102.png
```

Additional datasets can be placed in `dataset/<dataset-name>/{train,test}/<class_name>/...`; the training page lets you select a detected dataset. Keep original images from each source split together: split images into train/test **before** creating any augmented derivatives, and never put derivatives of test images into training. The app verifies matching class directories and supported images. It does not import detection datasets' original annotation formats automatically; image-level classification uses the class folder name.

If a dataset has not been installed, the upload and transformation pages still work. The optional illustrative demo is procedurally drawn and explicitly labeled; it is not a sample from SIXray, OPIXray, PIDray, or any real baggage scan. No dataset or model accuracy is fabricated.

## Project layout

```text
app.py
requirements.txt
README.md
modules/
  preprocessing.py
  augmentation.py
  generative_ai.py
  classifier.py
  evaluation.py
  storage.py
dataset/train/
dataset/test/
generated/
models/
reports/
assets/
utils/
```

Uploaded images and YOLO sidecars, generated PNGs, model artifacts, and reports are written locally under `assets/uploads/`, `assets/annotations/`, `generated/`, `models/`, and `reports/`. The latest models are `models/baseline_cnn.keras` and `models/augmented_cnn.keras`; versioned copies are stored under `models/experiments/`. Actual metrics are included only after a successful test-set evaluation.
