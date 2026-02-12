import os
import matplotlib.pyplot as plt

from metrics import center_error_norm
from helpers import get_gt_circles
from splits import train, val, test
from helpers import read_manual_results

participants = train + val + test



data_path = "./data/processed_data"

all_pp = []
labels = []

for par in participants:
    par_path = os.path.join(data_path, par)

    manual_file_path = os.path.join(par_path, "normalized_results_manual.txt")
    df = read_manual_results(manual_file_path)

    pp = []
    for i in range(len(df)):
        curr = df.iloc[i]

        _, r_gt, g_gt = get_gt_circles(curr)
        val = center_error_norm(r_gt, g_gt)
        pp.append(val)

    all_pp.append(pp)
    labels.append(par)

# Box plot: one box per participant
plt.figure(figsize=(12, 5))
plt.boxplot(all_pp, tick_labels=labels, showfliers=False)
plt.xticks(rotation=60, ha="right")
plt.ylabel("Center error (normalized units)")
plt.title("Center error distribution per participant")
plt.tight_layout()
plt.show()
