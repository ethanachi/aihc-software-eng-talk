import ml_library as ml

## Load the data, which has the following format:
# name  age  city   state   smoker

class Datapoint:
  def __init__(self, name, age, city, state, label):
    self.name = name
    self.age = int(age)
    self.city = city
    self.state = state
    self.label = (1 if label == "smoking" else 0)
    # preprocessing can be done away from our dataloading function
    # makes it easier to edit in future

  def get_data(self):
    return self.name, self.age, self.city, self.state

  def get_label(self):
    return self.label

class Dataset:
  def __init__(self, fname):
    self.data = self.load_data(fname)

  def load_data(self, fname):
    with open(fname, 'r') as f:
      train = f.readlines()
      train = [line.strip() for line in train]
      train = [line.split('\t') for line in train]

      out = []

      for (name, age, city, state, label) in train:
       if int(age) > 18 and state not in ['Alaska', 'Wyoming']:
          out.append(Datapoint(name, age, city, state, label))

    return out

  def get_data(self):
    return self.data

  def get_data_batch(self):
    return [x.get_data() for x in self.data]

  def get_label_batch(self):
    return [x.get_label() for x in self.data]

def train_classifier(train, dev, seed):
  learning_rate = 0.005
  classifier = ml.Classifier(dim=4, lr=learning_rate, seed=seed)
  last_acc = None
  for i in range(100):   # number of epochs
    for example in train.get_data():
      classifier.learn(example.get_data(), example.get_label())
    current_acc = classifier.calculate_acc(dev.get_data_batch(), dev.get_label_batch())
    if last_acc and last_acc > current_acc - 0.001:
      break
    last_acc = current_acc

  return classifier, current_acc

def report_results(classifier, dev):
  dev_predictions = classifier.predict_batch(dev.get_data_batch())
  num_printed = 0
  for data, label, prediction in zip(dev.get_data_batch(), dev.get_label_batch(), dev_predictions):
    print(data[0] + '\t' + str(data[1]) + '\t' + data[2] + '\t' + data[3] +
          '\t' + str(label) + '\t' + str(prediction), ((prediction > 0.5) == label))
    num_printed += 1
    if num_printed == 5: break

train = Dataset('train_data.txt')
dev = Dataset('dev_data.txt')

accs = []
for i in range(5):
  classifier, acc = train_classifier(train, dev, seed=i)
  accs.append(acc)
  report_results(classifier, dev)
