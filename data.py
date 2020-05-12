class Datapoint:
  def __init__(self, name, age, city, state, label):
    ## Load the data, which has the following format:
    # name  age  city   state   smoker
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

# ignore what's below for now!

class Dataset_c:
  def __init__(self, args, fname):
    self.args = args
    self.data = self.load_data(fname)

  def load_data(self, fname):
    with open(fname, 'r') as f:
      train = f.readlines()
      train = [line.strip() for line in train]
      train = [line.split('\t') for line in train]

      out = []

      for (name, age, city, state, label) in train:
       if int(age) > self.args['min_age'] and state not in self.args['illegal_states']:
          out.append(Datapoint(name, age, city, state, label))

    return out

  def get_data(self):
    return self.data

  def get_data_batch(self):
    return [x.get_data() for x in self.data]

  def get_label_batch(self):
    return [x.get_label() for x in self.data]
