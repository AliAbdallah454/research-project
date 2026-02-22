import os, glob, shutil, re
src = r"../data/wetcat_subset/Segmentation_Masks"
dst = r"../data/wetcat_prepared/0/binary_annotations/seq1"

os.makedirs(dst, exist_ok=True)

for f in glob.glob(os.path.join(src,"*")):
  base= os.path.basename(f)
  if ".csv" in base.lower():
    continue

  new = re.sub(r"_instruments_mask\.(png|jpg|jpeg)$", r"_class1.png", base, flags=re.IGNORECASE)
  if new == base:
    root, ext = os.path.splitext(base)
    new = f"{root}_class1.png"

  shutil.copy2(f, os.path.join(dst,new))
print("renamed masks:", sorted(os.listdir(dst))[:10])

