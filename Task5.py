import os
import glob

from Task1 import read_resize_normalize_image
from Task2 import load_all_filters, filter_image
from Task3 import relu_and_gap
from Task4 import load_fc, classify_vector


def classify_image(image_path, weights, biases, wfc, bfc):
    x = read_resize_normalize_image(image_path)
    y = filter_image(x, weights, biases)
    o = relu_and_gap(y)
    p = classify_vector(o, wfc, bfc)

    return p


def evaluate(data_folder, weights_folder, threshold=0.5):
    weights, biases = load_all_filters(weights_folder)
    wfc, bfc = load_fc(weights_folder)

    hotdog_paths = sorted(glob.glob(os.path.join(data_folder, "hotdog", "*.jpg")))
    pancake_paths = sorted(glob.glob(os.path.join(data_folder, "pancakes", "*.jpg")))

    correct_hotdogs = 0
    correct_pancakes = 0

    for path in hotdog_paths:
        p = classify_image(path, weights, biases, wfc, bfc)

        if p < threshold:
            correct_hotdogs += 1

    for path in pancake_paths:
        p = classify_image(path, weights, biases, wfc, bfc)

        if p >= threshold:
            correct_pancakes += 1

    total_correct = correct_hotdogs + correct_pancakes
    total_images = len(hotdog_paths) + len(pancake_paths)
    accuracy = total_correct / total_images

    print("Correct hotdogs:", correct_hotdogs, "/", len(hotdog_paths))
    print("Correct pancakes:", correct_pancakes, "/", len(pancake_paths))
    print("Accuracy:", accuracy)

    return accuracy