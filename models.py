import ml_library as ml

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

# ignore what's below for now!

def train_classifier_c(args, train, dev, seed):
  learning_rate = args['lr']
  classifier = ml.Classifier(dim=4, lr=learning_rate, seed=seed)
  last_acc = None
  for i in range(args['num_epochs']):   # number of epochs
    for example in train.get_data():
      classifier.learn(example.get_data(), example.get_label())
    current_acc = classifier.calculate_acc(dev.get_data_batch(), dev.get_label_batch())
    if last_acc and last_acc > current_acc - 0.001:
      break
    last_acc = current_acc

  return classifier, current_acc
