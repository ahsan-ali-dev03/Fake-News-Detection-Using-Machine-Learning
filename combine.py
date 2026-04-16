import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake['label'] = 1
true['label'] = 0

data = pd.concat([fake, true])

data.to_csv("dataset.csv", index=False)

print("Dataset created ✅")