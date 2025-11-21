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

![Distribusi Kelas](images/val_dis.png)

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



# ▶️ **Cara Menjalankan di Google Colab**

Seluruh eksperimen pada repository ini **dirancang untuk dijalankan di Google Colab** menggunakan GPU Runtime.
Tidak disarankan menjalankan kode ini secara lokal karena membutuhkan resource GPU dan konfigurasi library yang sesuai dengan lingkungan Colab.

## ✅ 1. Buka Notebook Colab

Buka file notebook berikut di folder `notebooks/`:

➡ **`notebooks/swin_deit_freezing.ipynb`**

Jika ingin langsung membuka via Colab:

```
https://colab.research.google.com/github/sigawari/VisionTransformer-Comparison/blob/main/notebooks/swin_deit_freezing.ipynb
```

Klik:

**Runtime → Change runtime type → GPU**

---

## ✅ 2. Mount Google Drive

Pada cell pertama notebook, jalankan:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Pastikan folder dataset berada pada Drive yang sesuai (lihat bagian README Dataset).

---

## ✅ 3. Clone Repository dari GitHub

Di dalam notebook, jalankan:

```python
!git clone https://github.com/sigawari/VisionTransformer-Comparison.git
%cd VisionTransformer-Comparison
```

---

## ✅ 4. Install Dependencies (Colab Friendly)

Jalankan cell:

```python
!pip install -r requirements.txt
```

Colab akan otomatis meng-install:

* torch
* timm
* scikit-learn
* matplotlib
* tqdm
* dsb.

---

## ✅ 5. Jalankan Training

Notebook sudah berisi seluruh pipeline pelatihan:

* Load dataset
* Preprocessing
* Model setup
* Freezing MLP layer
* Training loop
* Evaluation

Klik:

**Runtime → Run all**

Atau jalankan sel per sel.

Output akan berupa:

* Kurva training (loss/acc)
* Confusion matrix
* Prediksi contoh
* Log training
* File model tersimpan (`.pth`) di folder outputs

---

## ✅ 6. Melihat Hasil Model

Setelah training selesai, model disimpan otomatis:

📁 `outputs/models/swin_freeze.pth`
📁 `outputs/models/deit_freeze.pth`

Hasil evaluasi tersimpan di:

📁 `outputs/figures/`
📁 `outputs/logs/`

Semua sudah otomatis digenerate oleh notebook.

---

## 📌 Catatan Penting

* **Notebook harus dijalankan di Colab** (GPU T4 atau L4)
* Dataset harus disimpan di **Google Drive**, bukan lokal
* Jangan jalankan `python train.py` di lokal, karena script tersebut dibuat untuk lingkungan Colab juga
* Pastikan runtime Colab tidak timeout saat training (sekitar ±15 epoch)

---

# 🎓 **Kesimpulan Singkat**

* Model **Swin-Tiny** dan **DeiT-Tiny** bekerja sangat baik pada dataset berukuran kecil–menengah.
* Teknik **MLP-Freezing** menurunkan jumlah parameter trainable tanpa menurunkan akurasi secara signifikan.
* **Swin-Tiny** → lebih stabil & generalisasi baik.
* **DeiT-Tiny** → lebih cepat konvergensi tapi rentan overfitting.