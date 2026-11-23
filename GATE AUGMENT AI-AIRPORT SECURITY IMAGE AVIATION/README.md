# GateAugment AI

GateAugment AI is a local Streamlit research workstation for baggage X-ray image processing and classifier robustness experiments. It includes image/video scan processing, webcam-still demonstration mode, persistent scan history in SQLite, advanced annotation-aware augmentation, optional Diffusers image-to-image synthesis, optional custom X-ray YOLO detection, CNN training, analytics, and CSV/JSON/PDF/image downloads.

> **Research-only:** This application is not an operational airport screening system. Review every generated image and its annotations. A generic pretrained diffusion model is not X-ray-domain validated and can change security-relevant content.

## Requirements and installation

- Python 3.10–3.12 recommended.
- A CUDA-capable GPU is optional for diffusion and TensorFlow acceleration.
- Diffusion requires model weights and may download a large model the first time.
- Custom object detection requires Ultralytics and trusted X-ray-trained weights.

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

Optional custom object detection:

```powershell
pip install ultralytics
```

Only use trusted custom detector weights trained for a relevant X-ray dataset under `models/detection/`. No default COCO weights are loaded or represented as a security detector.

## Run

```powershell
streamlit run app.py
```

## Workstations and workflow

1. **Security Intelligence Dashboard:** project metrics, actual model status/accuracy, analytics charts, scan history, and optional component status.
2. **Live Scanner:** process uploaded images, sample frames from uploaded videos, or capture a regular webcam still in explicitly labeled demonstration mode. Run grayscale, CLAHE enhancement, edge detection, and sharpening; scan history is recorded in SQLite.
3. **Image Upload / Image Intelligence:** validate and persist JPG/JPEG/PNG uploads, inspect metadata and image statistics, and optionally run a trained image-level CNN.
4. **Advanced Augmentation:** use Albumentations/OpenCV rotation, flips, brightness/contrast, scale, translation, crop, controlled occlusion, Gaussian noise, blur, and overlap. Run a batch across uploaded images and export images plus annotation sidecars.
5. **Generative AI Studio:** a genuine Diffusers `StableDiffusionImg2ImgPipeline` can generate variations when PyTorch, Diffusers, suitable weights, and hardware are available. Generic pretrained weights are not X-ray-domain validated. An unavailable model may use a clearly labeled traditional fallback, never mislabeled as synthetic generation.
6. **Object Detection:** optional Ultralytics integration loads only custom local model weights from `models/detection/`. No default COCO model or threat-detection claim is used.
7. **Model Training / Evaluation / Results Comparison:** train baseline and Keras-augmented CNNs and evaluate both on the same untouched test set. View accuracy, macro precision/recall/F1, classification reports, confusion matrices, and training/validation curves.
8. **Smart Analytics / Reports and History / Settings:** inspect activity and scan metrics, see installed integrations, and export JSON, CSV, PDF, and generated-image ZIP reports.

The scanner does not connect to airport screening hardware or a live airport feed. Video is sampled to a bounded number of frames for local review. Webcam mode captures an ordinary camera still only. Image-enhancement outputs are not threat predictions.

## Persistent image upload and annotations

Uploaded images are validated and saved under `assets/uploads/`. The project stores metadata, generated-image provenance, recent activity, and experiment summaries in `reports/project_state.json`; SQLite scan/event history is stored at `reports/gateaugment.sqlite3`. Upload one or more matching YOLO `.txt` files to preserve boxes through geometric transformations. Each annotation file must share its image's filename stem and contain rows in this format:

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
  scanner.py
  database.py
  detection.py
  reports.py
dataset/train/
dataset/test/
generated/
models/
reports/
assets/
utils/
```

Uploaded images and YOLO sidecars, generated PNGs, scan results, model artifacts, and reports are written locally under `assets/uploads/`, `assets/annotations/`, `generated/`, `models/`, and `reports/`. The SQLite scanner/event log is `reports/gateaugment.sqlite3`. The latest models are `models/baseline_cnn.keras` and `models/augmented_cnn.keras`; versioned copies are stored under `models/experiments/`. Actual metrics are included only after a successful test-set evaluation.
