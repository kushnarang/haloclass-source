import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/Users/kushnarang/Downloads/haloclass-mutation-section-aug-12 - kde-aug-27.csv")

data['Experimental'] = data['experimental'].apply(lambda x: 1 if x == "increase" else 0)

data["Protein"] = data['protein']

sns.kdeplot(x=data['delta'], y=data['Experimental'], fill=True, bw_adjust=0.5)
sns.scatterplot(x=data['delta'], y=data['Experimental'], hue=data['Protein'])

# Add title and labels
plt.title('True vs. predicted mutant effect on salt tolerance')
plt.xlabel('Predicted delta (HaloClass)')
plt.ylabel('True change')
# Display the plot
fig = plt.gcf()
fig.set_size_inches(8,4, forward=True)
plt.savefig("publication-figures/sf5.png", dpi=300)