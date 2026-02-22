from dataset_wetcat import WetCatDataset

ds = WetCatDataset(
    data_root_dir="projects/surgicalsam-wetcat/data/wetcat_prepared",
    vit_mode="h",
    seq="seq1"
)

print("Dataset size:", len(ds))

feat, name, cls_id, mask, emb = ds[0]

print("Feature shape:", feat.shape)
print("Mask shape:", mask.shape)
print("Embedding shape:", emb.shape)
print("Class ID:", cls_id)
print("Name:", name)