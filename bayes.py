import os
import sys
import argparse
import logging
import math
def calculate_likelihood(x, mean, std_dev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std_dev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent
def calculate_mean(data):
    return sum(data) / len(data)

def calculate_std_dev(data, mean):
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)
def training(instances, labels):
    summaries = {}
    class_counts = {}
    
    for instance, label in zip(instances, labels):
        if label not in summaries:
            summaries[label] = []
            class_counts[label] = 0
        
        class_counts[label] += 1
        for i, attribute in enumerate(instance):
            if len(summaries[label]) <= i:
                summaries[label].append([])
            summaries[label][i].append(attribute)
    
    for label, summary in summaries.items():
        summaries[label] = [(calculate_mean(attribute), calculate_std_dev(attribute, calculate_mean(attribute)))
                             for attribute in summary]
    
    prior_probabilities = {label: count / len(labels) for label, count in class_counts.items()}
    
    return summaries, prior_probabilities

def predict(instance, parameters):
    summaries, prior_probabilities = parameters  # 매개변수로 받은 parameters를 summaries와 prior_probabilities로 분리
    probabilities = {}
    for label, class_summaries in summaries.items():
        probabilities[label] = prior_probabilities[label]
        for i in range(len(class_summaries)):
            mean, std_dev = class_summaries[i]
            x = instance[i]
            probabilities[label] *= calculate_likelihood(x, mean, std_dev)
    return max(probabilities, key=probabilities.get)

def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    # precision
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp), 2) * 100

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100
    f1_score = (2 * precision * recall) / (precision + recall)
    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))
    logging.info("f1 score: {}%".format(f1_score))

def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()  # 첫 번째 줄은 헤더이므로 무시합니다.
        for line in f:
            tmp = line.strip().split(", ")[1:]  # 첫 번째 열을 제외한 나머지 데이터만 사용합니다.
            # 데이터 변환
            tmp = [float(val) if i != len(tmp) - 1 else int(val) for i, val in enumerate(tmp)]
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels
"""
def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()  # 첫 번째 줄은 헤더이므로 무시합니다.
        for line in f:
            data = line.strip().split(", ")
            # 온도의 최대값, 습도의 평균값, 파워 소비 특성만 선택하여 데이터 처리
            selected_data = [float(data[2]), float(data[3]), float(data[5]), float(data[7])]
            label = int(data[-1])
            instances.append(selected_data)
            labels.append(label)
    return instances, labels
"""
def run(train_file, test_file):
    # training phase
    instances, labels = load_raw_data(train_file)
    logging.debug("instances: {}".format(instances))
    logging.debug("labels: {}".format(labels))
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file)
    predictions = []
    for instance in instances:
        result = predict(instance, parameters)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)
    
    # report
    report(predictions, labels)

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True, metavar="<file path to the training dataset>", help="File path of the training dataset", default="training.csv")
    parser.add_argument("-u", "--testing", required=True, metavar="<file path to the testing dataset>", help="File path of the testing dataset", default="testing.csv")
    parser.add_argument("-l", "--log", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")

    args = parser.parse_args()
    return args

def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    run(args.training, args.testing)

if __name__ == "__main__":
    main()
