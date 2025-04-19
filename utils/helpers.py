from sklearn.model_selection import train_test_split


def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df["preprocess_text"]
    y = (
        df[target_column]
        .str.strip()
        .str.lower()
        .map({"positive": 2, "neutral": 1, "negative": 0})
    )
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test
