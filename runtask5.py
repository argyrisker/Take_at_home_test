from Task5 import evaluate


if __name__ == "__main__":
    data_folder = "/projects/mai/se_mai/users/kjzj855_argyrios/project/TakeHomeTest/data"
    weights_folder = "/projects/mai/se_mai/users/kjzj855_argyrios/project/TakeHomeTest"

    evaluate(data_folder, weights_folder, threshold=0.5)