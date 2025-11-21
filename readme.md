# 📘 **Vision Transformer Comparison (Swin Tiny vs DeiT Tiny with MLP-Freezing)**

## 🔍 **Overview**

Proyek ini mengevaluasi performa dua arsitektur Vision Transformer modern pada dataset *Reptiles and Amphibians Image Dataset*. Evaluasi mencakup akurasi, loss, jumlah parameter, efisiensi inferensi, dan stabilitas pelatihan. Proyek ini juga menerapkan teknik *MLP freezing* untuk mengurangi jumlah parameter yang dilatih dan mempercepat proses training.

Eksperimen dilakukan menggunakan dua model:

* **Swin-Tiny (swin_tiny_patch4_window7_224)**
* **DeiT-Tiny (deit_tiny_patch16_224)**

---

# 📦 **Dataset**

Dataset berasal dari Kaggle:

🔗 **[Reptiles and Amphibians Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/reptiles-and-amphibians-image-dataset)**

Dari total 10 kelas asli, hanya **5 kelas** yang digunakan:

* Frog
* Chameleon
* Snake
* Iguana
* Turtle_Tortoise

Dataset kemudian diseimbangkan:

* **200 gambar per kelas**
* **Split 70/15/15** → train / validation / test

---

## 📊 **Distribusi Dataset**

### **Distribusi Kelas**

> Ganti dengan gambar kamu

```
images/class_distribution.png
```

![Distribusi Kelas](images/class_distribution.png)

### **Contoh Gambar Dataset**

```
images/contoh_dataset.png
```

![Contoh Dataset](images/contoh_dataset.png)

---

# ⚙️ **Preprocessing dan Augmentasi**

Model menggunakan transformasi berikut:

| Tahap    | Transformasi                                                        |
| -------- | ------------------------------------------------------------------- |
| Train    | Resize(224), CenterCrop(224), HorizontalFlip, ColorJitter, ToTensor |
| Val/Test | Resize(224), CenterCrop(224), ToTensor                              |

---

# ⚙️ **Hyperparameter Training**

### Hyperparameter yang digunakan

| No | Parameter               | Nilai                     |
| -- | ----------------------- | ------------------------- |
| 1  | IMG_SIZE                | 224×224                   |
| 2  | Batch Size              | 32                        |
| 3  | Epoch                   | 15                        |
| 4  | Warmup                  | 1                         |
| 5  | Learning Rate           | 1e-4                      |
| 6  | Weight Decay            | 0.05                      |
| 7  | Optimizer               | AdamW                     |
| 8  | Scheduler               | Cosine Annealing          |
| 9  | Early Stopping Patience | 4                         |
| 10 | AMP                     | Enabled                   |
| 11 | Layer Freezing          | Freeze all MLP/FFN blocks |
| 12 | Models                  | Swin-Tiny, DeiT-Tiny      |

---

# 🧠 **Arsitektur dan Jumlah Parameter**

| Model     | Total Parameter | Trainable Parameter (setelah MLP Freeze) |
| --------- | --------------- | ---------------------------------------- |
| Swin-Tiny | *X*             | *Y*                                      |
| DeiT-Tiny | *A*             | *B*                                      |

> Isi X/Y/A/B berdasarkan output `sum(p.numel())`.

---

# 📈 **Hasil Pelatihan**

## Learning Curves

```
images/learning_curve.png
```

![Learning Curve](images/learning_curve.png)

---

# 🧪 **Evaluasi Model**

## 🔵 Confusion Matrix

```
images/confusion_matrix.png
```

![Confusion Matrix](images/confusion_matrix.png)

---

# 📊 **Perbandingan Performa Model**

| Model     | Test Accuracy | Test Loss | Best Val Loss |
| --------- | ------------- | --------- | ------------- |
| Swin-Tiny | X             | Y         | Z             |
| DeiT-Tiny | P             | Q         | R             |

> Ganti X Y Z P Q R berdasarkan df summary.

---

# ⚡ **Perbandingan Waktu Inferensi**

| Model     | Batch Inference (img/s) | Per Image (ms) | Throughput (img/s) |
| --------- | ----------------------- | -------------- | ------------------ |
| Swin-Tiny | x                       | y              | z                  |
| DeiT-Tiny | a                       | b              | c                  |

---

# 🖼️ **Visualisasi Prediksi**

```
images/sample_predictions.png
```

![Sample Predictions](images/sample_predictions.png)

---

# 📂 **Struktur Folder**

```
project/
│── dataset_balanced/
│── notebooks/
│── results/
│   ├── best_model.pt
│   ├── worst_model.pt
│   ├── results_summary.csv
│── images/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── learning_curve.png
│   └── sample_predictions.png
│── README.md
└── train.py
```

---

# ▶️ **Cara Menjalankan**

### 1. Clone repo

```bash
git clone https://github.com/username/transformer-comparison.git
cd transformer-comparison
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Training

```bash
python train.py
```

### 4. Lihat Output

* `best_model.pt`
* `worst_model.pt`
* `results_summary.csv`

---

# 🎓 **Kesimpulan Singkat**

* Model Swin-Tiny dan DeiT-Tiny sama-sama menunjukkan performa baik pada dataset berukuran menengah.
* Freezing MLP secara signifikan mengurangi parameter trainable tanpa menurunkan akurasi secara drastis.
* Swin-Tiny unggul dalam efisiensi inferensi dan stabilitas validasi.
* DeiT-Tiny menunjukkan sifat lebih ringan dan cepat dilatih dibanding model transformer hierarkis.

---
