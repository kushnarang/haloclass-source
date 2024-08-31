import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("publication-datasets/sf/arch_sel.csv")
df = df.drop(["iter"], axis=1)

dfm = pd.melt(df, id_vars="model", var_name="metric", value_name="scores")
dfm["Model Type"] = dfm["model"].apply(lambda s: s.replace(" ", "\n"))

g = sns.catplot(
    data=dfm, kind="bar", errorbar="sd",
    x="Model Type", y="scores", hue="metric",
)

plt.ylim(bottom=0.86, top=1.0)

fig = plt.gcf()
fig.set_size_inches(10,5, forward=True)
fig.savefig('publication-figures/sf1.png', dpi=300)