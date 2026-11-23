# GateAugment AI

GateAugment AI is a local Streamlit research application for baggage X-ray image augmentation and CNN robustness experiments. It supports image upload, traditional image transformations, an optional pretrained Diffusers image-to-image pipeline, TensorFlow/Keras classifier training, held-out test evaluation, visual comparisons, and JSON/image downloads.

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
4. **Model training:** train a baseline CNN and a second CNN with Keras random flip/rotation/contrast layers applied to training samples. The original train directory is split into training and validation subsets before training; the separate test directory is not used for fitting or augmentation. Both models are evaluated on that same test set.
5. **Evaluation, comparison, and reports:** view actual metrics and confusion matrices, compare generated images, and export JSON metrics/provenance and a ZIP of generated PNGs.

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

Keep original images from each source split together: split images into train/test **before** creating any augmented derivatives, and never put derivatives of test images into training. The app verifies that train/test class folder names match and that each class contains supported images. The default trainer uses whole-image class labels; it does not parse object detection or bounding-box annotations. If annotations are needed, transformations and annotation coordinates must be transformed together in a dataset-specific pipeline before training.

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
dataset/train/
dataset/test/
generated/
models/
reports/
assets/
utils/
```

Generated PNGs and model artifacts are written locally under `generated/` and `models/`. The report includes actual model metrics only after a successful evaluation run.
