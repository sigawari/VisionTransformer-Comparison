# Membuat folder baru berisi 200 gambar/kelas lalu split 70/15/15
import os, random, shutil
from glob import glob

src = "dataset"
dst = "dataset_balanced"
os.makedirs(dst, exist_ok=True)

TARGET_PER_CLASS = 200 

for cls in sorted(os.listdir(src)):
    src_cls = os.path.join(src, cls)
    if not os.path.isdir(src_cls): 
        continue
    imgs = [p for p in glob(os.path.join(src_cls, "*")) if os.path.isfile(p)]
    random.shuffle(imgs)
    pick = imgs[:TARGET_PER_CLASS]

    # Buat split
    n = len(pick)
    n_train = int(0.70*n)
    n_val   = int(0.15*n)

    splits = [
        ("train", pick[:n_train]),
        ("val",   pick[n_train:n_train+n_val]),
        ("test",  pick[n_train+n_val:]),
    ]

    for split_name, files in splits:
        out_dir = os.path.join(dst, split_name, cls)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            shutil.copy2(f, out_dir)

print("Selesai membuat dataset_balanced/{train,val,test}")
