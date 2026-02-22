from pathlib import Path

root = Path("../data/wetcat_prepared/0")

images = sorted((root / "images/seq1").glob("*.jpg"))
masks  = sorted((root / "binary_annotations/seq1").glob("*.png"))

print("Images:", len(images))
print("Masks:", len(masks))

print("Example image:", images[0].name)
print("Example mask :", masks[0].name)