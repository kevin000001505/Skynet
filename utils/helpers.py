from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df["preprocess_text"]
    if isinstance(df[target_column].iloc[0], int):
        y = df[target_column]
    else:
        y = df[target_column].astype(str).str.strip().str.lower()
    y_encoded = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test, le
