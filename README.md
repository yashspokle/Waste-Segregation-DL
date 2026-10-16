# WasteWise --- Waste Segregation Using Deep Learning

WasteWise is an academic waste-classification project that predicts one
of six waste categories from an uploaded image or a webcam capture. The
Streamlit interface displays the predicted category, model score, and
basic disposal guidance.


## Live Demo

**Try WasteWise here:** [https://waste-segregation.streamlit.app/](https://waste-segregation.streamlit.app/)

The deployed Streamlit app supports waste-image classification. Open the link to try image upload and, where your browser allows it, camera capture.

## Project highlights

-   Six-class image classification: cardboard, glass, metal, paper,
    plastic, and trash.
-   Two model experiments: MobileNetV3 and the model saved under
    `efficientnet_lite`.
-   Evaluation on a held-out test set of 380 images.
-   Confusion matrices and misclassified-image exports.
-   Inference latency and FPS measurements.
-   Streamlit UI supporting image upload and browser-camera capture.

> **Architecture naming note:** The results folder is named
> `efficientnet_lite`. Confirm the architecture used in `train.py`
> before describing it as EfficientNet-Lite. If the code uses
> `tf.keras.applications.EfficientNetB0`, call it **EfficientNetB0** in
> the report and repository; EfficientNetB0 and EfficientNet-Lite are
> different architectures.

## Dataset

The project uses TrashNet-style images organized into six class folders:

``` text
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

-   Total images found: **2,527**
-   Test images evaluated: **380**
-   Input size: **224 × 224 RGB**

The test split is intended to represent 15% of the dataset. Keep the
same split logic and random seed for reproducible comparisons. If
`evaluate.py` reconstructs the split independently, ensure its class
ordering, file ordering, seed, and split logic match `train.py`.

## Model evaluation results

The following values are from the recorded evaluation run:

## 📊 Model Evaluation

![Model Evaluation Results](results/plots/model_evaluation.png)

### Results summary

-   The model saved under `efficientnet_lite` had the higher test
    accuracy: **90.53%**.
-   Its Macro F1-score was **89.47%**, compared with **82.37%** for
    MobileNetV3.
-   It made **36** incorrect predictions, versus **58** for MobileNetV3
    --- 22 fewer errors on this test set.
-   MobileNetV3 was faster in this CPU evaluation: **5.597 ms/image**
    and **178.68 FPS**.
-   The latency and FPS values are measured results from this run and
    may vary by machine, TensorFlow version, batch size, and runtime
    settings.

## Application features

### Upload Image

1.  Select a JPG, JPEG, or PNG image.
2.  Preview the image.
3.  Select **Detect Waste**.
4.  View the predicted class, model score, and disposal tip.

### Camera Scan

1.  Open **Camera Scan** in the app.
2.  Allow browser camera permission.
3.  Capture an image.
4.  Select **Detect Captured Waste**.

Camera capture typically requires `localhost` during local development
or HTTPS when deployed.

## Repository structure

``` text
Waste-Segregation-DL/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── mobilenetv3/
│   │   └── best_model.keras
│   └── efficientnet_lite/
│       └── best_model.keras
├── results/
│   ├── metrics/
│   │   ├── comparison.json
│   │   └── evaluation_results.json
│   ├── plots/
│   │   ├── mobilenetv3_confusion_matrix.png
│   │   └── efficientnet_lite_confusion_matrix.png
│   └── misclassified/
└── README.md
```

Actual files may differ depending on which training and evaluation
scripts have been run.

## Requirements

-   Python 3.10 or a Python version compatible with the TensorFlow
    version installed in your environment
-   TensorFlow
-   Streamlit
-   NumPy
-   Pillow
-   scikit-learn
-   Matplotlib
-   Seaborn

Install dependencies inside your virtual environment:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `.venv` already exists, activate it instead of recreating it.

## Run the application locally

From the project root:

``` powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Train and evaluate

Run these only when needed; training can take time on a CPU:

``` powershell
python train.py
python evaluate.py
```

The evaluation script saves metrics, confusion matrices, and
misclassified images under `results/`.

## Push to a new GitHub repository

### 1. Create a repository on GitHub

Create a **new empty repository** on GitHub. Do not initialize it with a
README, license, or `.gitignore` if you plan to push this existing local
project directly.

### 2. Add a `.gitignore`

Create a `.gitignore` file in the project root. Suggested contents:

``` gitignore
.venv/
__pycache__/
*.py[cod]
.streamlit/secrets.toml
.env
.env.*
```

Decide whether to commit the dataset and trained models before pushing.
They may be large. GitHub has file-size limits, and model/data files can
make the repository difficult to clone. If they are too large, use Git
LFS or publish clear instructions for obtaining the dataset and model
files separately.

### 3. Initialize and push

Replace `YOUR-USERNAME` and `YOUR-NEW-REPO` with your GitHub username
and repository name:

``` powershell
cd D:\Waste-Segregation-DL
git init
git add .
git status
git commit -m "Initial commit: waste segregation deep learning project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-NEW-REPO.git
git push -u origin main
```

If this folder already has a Git repository, check `git status` and
`git remote -v` first. Do not run `git remote add origin` a second time
if `origin` already exists.

If Git prompts for authentication, complete the browser sign-in or use a
GitHub-supported authentication method. Do not put access tokens in the
repository.

## Deployment

The project is deployed on **Streamlit Community Cloud**:

- **Live app:** [https://waste-segregation.streamlit.app/](https://waste-segregation.streamlit.app/)

To update the deployed app, push changes to the connected GitHub repository. Streamlit Community Cloud typically rebuilds the app automatically when the configured branch is updated.

### Deployment setup and troubleshooting

A straightforward deployment setup is **Streamlit Community Cloud**:

1.  Push the project to GitHub.
2.  Open Streamlit Community Cloud and connect your GitHub account.
3.  Choose the repository, branch `main`, and main file path `app.py`.
4.  Deploy the app.
5.  Check dependency versions and model-file availability if deployment
    fails.

Before deployment: - Add a compatible, tested `requirements.txt` (pin
versions once your local environment works). - Ensure the model file is
available to the deployed app. Large `.keras` files may require Git LFS
or external storage, subject to hosting support and limits. - Do not
commit `.venv/`, credentials, secrets, or unnecessary raw data. - Test
webcam access on the hosted HTTPS URL; browser permission and hosting
limitations can affect camera capture. - If the model is too large or
startup is slow, consider storing model artifacts separately and loading
them at runtime.

## Limitations

-   The classifier predicts among the six categories represented in its
    training data; it may be unreliable on unfamiliar items, mixed
    waste, unusual lighting, or images containing multiple objects.
-   The displayed model score is not a calibrated guarantee of
    correctness.
-   Bin labels and disposal rules vary by municipality. The
    recommendations in this prototype are general guidance, not official
    local rules.
-   Test metrics apply to this test split and are not a guarantee of
    real-world performance.

## Future improvements

-   Add per-class precision, recall, and F1-score charts.
-   Review confusion matrices and representative misclassified samples.
-   Test on images from outside the training dataset.
-   Compare accuracy against model size and inference speed.
-   Add model/version metadata and a reproducible experiment
    configuration.
-   Validate disposal guidance against local municipal recycling rules.

## License

Add a license before public release if you intend others to reuse the
code. Also confirm the dataset's license and attribution requirements
before redistributing any images.
