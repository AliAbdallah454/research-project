import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from splits import train, val, test
from helpers import read_manual_results

participants = train + val + test

def collect_radius_ratios(participants, data_path="./data/processed_data"):
    rows = []
    for par in participants:
        par_path = os.path.join(data_path, par)
        manual_file_path = os.path.join(par_path, "normalized_results_manual.txt")

        df = read_manual_results(manual_file_path)

        z1 = pd.to_numeric(df["Radius Zone 1"], errors="coerce")
        z2 = pd.to_numeric(df["Radius Zone 2"], errors="coerce")
        ratio = (z2 / z1).replace([np.inf, -np.inf], np.nan).dropna()

        rows.append(pd.DataFrame({"participant": par, "ratio": ratio.values}))

    return pd.concat(rows, ignore_index=True)

ratios_df = collect_radius_ratios(participants, data_path="./data/processed_data")

participants_order = (
    ratios_df.groupby("participant")["ratio"].median().sort_values().index.tolist()
)

data_for_box = [
    ratios_df.loc[ratios_df["participant"] == p, "ratio"].values
    for p in participants_order
]

plt.figure(figsize=(12, 5))
plt.boxplot(data_for_box, tick_labels=participants_order, showfliers=False)
plt.xticks(rotation=60, ha="right")
plt.ylabel("Radius ratio (Zone2 / Zone1)")
plt.title("Radius ratio distribution per participant")
plt.tight_layout()
plt.show()