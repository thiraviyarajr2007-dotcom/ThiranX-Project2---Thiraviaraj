# 🦟 Malaria Cell Classification using CNN + ROC Analysis

A binary image classifier that detects whether a blood-smear cell image is **Parasitized** (malaria-infected) or **Uninfected**, built with a Convolutional Neural Network in **Keras**. Beyond plain accuracy, the project uses **confusion matrices, the ROC curve and AUC** to pick a good decision threshold instead of blindly using 0.5.

> **Credit:** The original code was written by **Sreenivas Bhattiprolu** ([Python for Microscopists](https://www.youtube.com/@DigitalSreeni)). Tutorial videos: [Part 1](https://youtu.be/vF21cC-8G1U) · [Part 2](https://youtu.be/Joh3LOaG8Q0). The author's licence note: *"Feel free to copy, I appreciate if you acknowledge Python for Microscopists."*

---

## 📌 Problem Statement

Given a cell image, predict whether it is infected with the malaria parasite.

| Label | Class        | Meaning of model output                  |
|-------|--------------|------------------------------------------|
| `0`   | Parasitized  | Probability close to **0** → infected    |
| `1`   | Uninfected   | Probability close to **1** → healthy     |

The network ends with a single **sigmoid** neuron that outputs the probability that the image is *uninfected*.

## 📊 Dataset

- **Source:** [NIH / NLM malaria cell images dataset](https://lhncbc.nlm.nih.gov/publication/pub9932)
- **Classes:** `Parasitized` and `Uninfected` (PNG images)
- **Expected folder layout:**

```
cell_images/
├── Parasitized/
│   └── *.png
└── Uninfected/
    └── *.png
```

> The dataset is **not** included in this repo. Download it and place it in `cell_images/` next to the script.

## 🧠 Approach

1. **Load & label** all `.png` images (Parasitized → 0, Uninfected → 1)
2. **Resize** every image to `150 × 150` RGB
3. **Split** into 80% train / 20% test (`random_state=0`)
4. **Normalize** pixel data with `keras.utils.normalize`
5. **Train** a CNN with `binary_crossentropy` loss and the `rmsprop` optimizer
6. **Evaluate** with accuracy, confusion matrix, ROC curve and AUC
7. **Tune the threshold** using the ROC curve

### Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D → ReLU → MaxPool | 32 filters, 3×3, pool 2×2 |
| Conv2D → ReLU → MaxPool | 32 filters, 3×3, `he_uniform` init, pool 2×2 |
| Conv2D → ReLU → MaxPool | 64 filters, 3×3, `he_uniform` init, pool 2×2 |
| Flatten | |
| Dense → ReLU | 64 units |
| Dropout | 0.5 |
| Dense → Sigmoid | 1 unit (binary output) |

**Training settings:** batch size 64 · 10 epochs · `shuffle=False` · validation on the test split.

## 📈 Evaluation & ROC Analysis

- **Accuracy:** the script's notes report about **82.5%** for a model trained for 300 epochs. Results for the default 10-epoch run will differ.
- **Confusion matrix:** computed at a custom threshold (`0.908`) to show how threshold choice changes false positives and false negatives.
- **ROC curve:** plots True Positive Rate vs False Positive Rate across all thresholds.
- **Optimal threshold:** found where `TPR − (1 − FPR)` is closest to zero, balancing sensitivity and specificity.
- **AUC:** area under the ROC curve, which measures how well the model separates the two classes.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- tensorflow / keras
- opencv-python
- pillow
- numpy
- pandas
- scikit-learn
- matplotlib

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install tensorflow opencv-python pillow numpy pandas scikit-learn matplotlib
```

### Run

1. Download the dataset and place it in `cell_images/`
2. Run the script:

```bash
python "Binery_Classifiaction_using_ROC_.py"
```

The script trains the model, saves it as `malaria_model_10epochs.h5`, plots loss and accuracy curves, then runs the ROC analysis.

## ⚠️ Known Issues & Tips

- **Model file mismatch:** training saves `malaria_model_10epochs.h5`, but the evaluation section loads `malaria_model.h5` (the author's pre-trained 300-epoch model). Either change the load path to `malaria_model_10epochs.h5` or supply your own `malaria_model.h5`.
- **Keras history keys:** newer Keras/TensorFlow versions use `history.history['accuracy']` and `['val_accuracy']` instead of `'acc'` and `'val_acc'`. Update these if you get a `KeyError`.
- **Image size:** the code uses `SIZE = 150`, although some comments mention 64×64. The code value is what applies.
- **Threshold:** the `0.908` threshold is specific to the author's trained model. Re-derive it from the ROC step for your own model.
- **Longer training:** more epochs (the author used 300) improves performance. Also try the `adam` optimizer.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

## 🙏 Acknowledgements

- **Sreenivas Bhattiprolu**, author of the original code and the *Python for Microscopists* series
- **U.S. National Library of Medicine (NLM / NIH)** for the malaria cell images dataset

## 📄 License

Code is shared under the author's permissive terms: free to copy and use, with acknowledgement to *Python for Microscopists*. The dataset follows the terms of the NLM / NIH source.
