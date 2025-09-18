from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ds = fetch_openml('mnist_784', as_frame=False)

x, x_test, y, y_test = train_test_split(
    ds.data, ds.target,
    test_size=0.2, random_state=42
)
