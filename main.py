import ml_library as ml

## Load the data, which has the following format:
# name  age  city   state   smoker


with open('train_data.txt', 'r') as f:
  train = f.readlines()
  for i in range(len(train)):
   train[i] = train[i].strip()
  for i in range(len(train)):
   train[i] = train[i].split('\t')

  train_data = []
  train_labels = []
  for datapoint in train:
   if int(datapoint[1]) > 18:   # only keep adults
    if datapoint[3] not in ['Alaska', 'Wyoming']:  # not legally allowed to perform experiment in these states
      if datapoint[4] == "smoking":
        smoking_label = 1
      else:
        smoking_label = 0
      train_data.append((datapoint[0], int(datapoint[1]), datapoint[2], datapoint[3]))
      train_labels.append(smoking_label)

with open('dev_data.txt', 'r') as f:
  dev = f.readlines()
  for i in range(len(dev)):
   dev[i] = dev[i].strip()
  for i in range(len(dev)):
   dev[i] = dev[i].split('\t')

  dev_data = []
  dev_labels = []
  for datapoint in dev:
    if int(datapoint[1]) > 18:   # only keep adults
      if datapoint[3] not in ['Alaska', 'Wyoming']:  # not legally allowed to perform experiment in these states
        if datapoint[4] == "smoking":
          smoking_label = 1
        else:
          smoking_label = 0
        dev_data.append((datapoint[0], int(datapoint[1]), datapoint[2], datapoint[3]))
        dev_labels.append(smoking_label)

## Whew, we're finished loading the data.  Let's train a classifier five times with different random seeds.
## We use the learn function to update our parameters

accs = []
for i in range(5):
  print("Seed", i)
  learning_rate = 0.005
  classifier = ml.Classifier(dim=4, lr=learning_rate, seed=i)
  last_acc = None
  for i in range(100):   # number of epochs
    for j in range(len(train_data)):
      classifier.learn(train_data[j], train_labels[j])
    current_acc = classifier.calculate_acc(dev_data, dev_labels)
    print(current_acc)
    if last_acc and last_acc > current_acc - 0.001:
      break
    last_acc = current_acc
  accs.append(current_acc)

  ## Finally, let's print out our predictions on the dev set:

  dev_predictions = classifier.predict_batch(dev_data)
  num_printed = 0
  for i in range(len(dev_data)):
    print(dev_data[i][0] + '\t' + str(dev_data[i][1]) + '\t' + dev_data[i][2] + '\t' + dev_data[i][3] +
          '\t' + str(dev_labels[i]) + '\t' + str(dev_predictions[i]), ((dev_predictions[i] > 0.5) == dev_labels[i]))
    num_printed += 1
    if num_printed == 5: break
