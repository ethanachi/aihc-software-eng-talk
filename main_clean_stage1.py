import ml_library as ml

## Load the data, which has the following format:
# name  age  city   state   smoker

def get_data(fname):
  with open(fname, 'r') as f:
    train = f.readlines()
    train = [line.strip() for line in train]
    train = [line.split('\t') for line in train]

    data = []
    labels = []

    for (name, age, city, state, label) in train:
     if int(age) > 18 and state not in ['Alaska', 'Wyoming']:
        data.append((name, int(age), city, state))
        labels.append(1 if label == "smoking" else 0)

  return data, labels

def train_classifier(train_data, train_labels, dev_data, dev_labels, seed):
  learning_rate = 0.005
  classifier = ml.Classifier(dim=4, lr=learning_rate, seed=seed)
  last_acc = None
  for i in range(100):   # number of epochs
    for data, label in zip(train_data, train_labels):
      classifier.learn(data, label)
    current_acc = classifier.calculate_acc(dev_data, dev_labels)
    print(current_acc)
    if last_acc and last_acc > current_acc - 0.001:
      break
    last_acc = current_acc

  return classifier, current_acc

def report_results(classifier, dev_data, dev_labels):
  dev_predictions = classifier.predict_batch(dev_data)
  num_printed = 0
  for data, label, prediction in zip(dev_data, dev_labels, dev_predictions):
    print(data[0] + '\t' + str(data[1]) + '\t' + data[2] + '\t' + data[3] +
          '\t' + str(label) + '\t' + str(prediction), ((prediction > 0.5) == label))
    num_printed += 1
    if num_printed == 5: break

train_data, train_labels = get_data('train_data.txt')
dev_data, dev_labels = get_data('dev_data.txt')

accs = []
for i in range(5):
  classifier, acc = train_classifier(train_data, train_labels,
                                     dev_data, dev_labels, seed=i)
  accs.append(acc)
  report_results(classifier, dev_data, dev_labels)
