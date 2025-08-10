# Importlibraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

#
print("First 5 rows of the dataset:")
print(df.head())

#
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

#--------------------------

#
print("\nBasic statistics:")
print(df.describe())

#
grouped_means = df.groupby('species').mean()
print("\nMean values by species:")
print(grouped_means)

#
print("\nInteresting findings:")
print("Iris-virginica has the highest average petal length and width.")
print("Iris-setosa has the smallest average measurements overall.")


#SetSeaborntheme
sns.set(style="whitegrid")

#
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title('Line Chart: Sepal and Petal Length Over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

#
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Bar Chart: Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()

# 
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True)
plt.title('Histogram: Sepal Width Distribution')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()
