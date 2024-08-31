import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("publication-datasets/sf/plm_sel.csv")
df = df.drop(["iter"], axis=1)

dfm = pd.melt(df, id_vars="model", var_name="metric", value_name="scores")
dfm["Embedding Source"] = dfm["model"]

g = sns.catplot(
    data=dfm, kind="bar",
    x="Embedding Source", y="scores", hue="metric",
)

plt.ylim(bottom=0.85, top=1.0)

fig = plt.gcf()
fig.set_size_inches(10,5, forward=True)

fig.savefig('publication-figures/sf2.png', dpi=300)
