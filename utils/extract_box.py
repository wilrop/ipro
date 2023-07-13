import pandas as pd


def extract_box(df):
    idxs = df.idxmax()
    for idx in idxs:
        print(f'Pareto point: {df.iloc[idx].to_numpy()}')
    print(f'Nadir: {df.min().to_numpy()}')


if __name__ == '__main__':
    df = pd.read_csv('udrl.csv')
    extract_box(df)
