from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

df = pd.read_csv("publication-datasets/sf/hyper_sel.csv", converters={'model': eval})
df = df.drop(["auroc", "acc", "auprc"], axis=1)

df["gamma"] = df["model"].apply(lambda x: x["gamma"])
df["C"] = df["model"].apply(lambda x: x["C"])

print(df["model"].apply(lambda x: x["C"]))

df = df.drop(["model"], axis=1)

dfs = list(df.groupby(["gamma", "C"]))

rows = []

import numpy as np, scipy.stats as st

for i, df in dfs:
    mccs = df["mcc"].values
    mean_value = mccs.mean()
    ci = st.t.interval(0.95, len(mccs)-1, loc=np.mean(mccs), scale=st.sem(mccs))
    print(mean_value, ci)

    rows.append({
        "mcc": mean_value,
        "gamma": df["gamma"].tolist()[0],
        "C": df["C"].tolist()[0]
    })

df = pd.DataFrame(rows)
dfp = df.pivot(columns="gamma", index="C", values="mcc")

sns.heatmap(dfp, annot=True, fmt=".4f")

fig = plt.gcf()
fig.set_size_inches(6.5,4, forward=True)
fig.savefig('publication-figures/sf4.png', dpi=300)
