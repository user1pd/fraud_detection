import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fraud_detection.dataset import TransactionData

data = TransactionData()
print(data.__dict__)
# data.load()



# print(data.transaction_path)
# print(data.identity_path)
# print(data.df)
# print(data.X_train)


# print(data.df.shape)
# print(data.df.head())
# print(data.df.columns.tolist())

