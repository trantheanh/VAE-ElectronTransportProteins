from model.vae import VAE
from utility.io import load_fold


def main():
    (train_X, train_Y), (test_X, test_Y) = load_fold(
        ["data/input.fold.train1.csv"],
        ["data/input.fold.test1.csv"]
    )

    print(train_X.shape)

    vae = VAE()

    # Train VAE
    vae.fit(train_X)

    # Sampling 1 data point from each data point in train set
    samples = vae.generate(X=train_X)

    return


if __name__ == "__main__":
    main()