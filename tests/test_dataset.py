import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pip install -e (Install project as a package)

from fraud_detection.dataset import TransactionData

def test_transaction_data_initialization():
    data = TransactionData()

    assert data.df is None
    assert data.X_train is None
    assert data.X_test is None
    assert data.y_train is None
    assert data.y_test is None


def test_load_data():
    data = TransactionData()
    data.load()

    assert data.df is not None
    assert len(data.df) > 0
    assert data._loaded is True
