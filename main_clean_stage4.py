import sys # to read command-line arguments

import ml_library as ml
import yaml

# our own code
import data
import models
import reporter



def __main__():
  with open(sys.argv[1], 'r') as yaml_file:
    args = yaml.load(yaml_file)

  train = data.Dataset_c(args, 'train_data.txt')
  dev = data.Dataset_c(args, 'dev_data.txt')

  accs = []
  for i in range(5):
    print("Seed", i)
    classifier, acc = models.train_classifier_c(args, train, dev, seed=i)
    accs.append(acc)
    reporter.report_results_c(args, classifier, dev)
    print()

if __name__ == "__main__":
  __main__()
