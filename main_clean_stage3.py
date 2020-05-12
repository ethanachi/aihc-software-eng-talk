import ml_library as ml

# our own code
import data
import models
import reporter



def __main__():
  train = data.Dataset('train_data.txt')
  dev = data.Dataset('dev_data.txt')

  accs = []
  for i in range(5):
    print("Seed", i)
    classifier, acc = models.train_classifier(train, dev, seed=i)
    accs.append(acc)
    reporter.report_results(classifier, dev)
    print()

if __name__ == "__main__":
  __main__()
