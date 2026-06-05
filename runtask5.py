from Task5 import evaluate


if __name__ == "__main__":
    data_folder = "/project/TakeHomeTest/data"
    weights_folder = "/project/TakeHomeTest"

    evaluate(data_folder, weights_folder, threshold=0.5)
