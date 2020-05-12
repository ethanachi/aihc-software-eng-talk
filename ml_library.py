"""A basic demonstration library that presents a logistic regression classifier in vanilla Python."""

import math
import random

STATES = {
  'Missouri': -2,
  'California': -1,
  'Wyoming': 0,
  'Alaska': 0,
  'New York': +1,
  'North Dakota': +2,
}

AGE_RANGE = [15, 50]

FIRST_NAMES = [
  "John",
  "Emily",
  "Marius",
  "Xin",
  "Anthony",
  "Ellie",
  "Rose",
  "Dorothy",
  "Xiaofeng",
  "Pranav"
]

LAST_NAMES = [
  "Smith",
  "Xiao",
  "Liu",
  "Rajpurkar",
  "Mellian",
  "'t Hooft",
  "Jackson"
]

CITIES = [
  "Fairfield",
  "Fair Oaks",
  "Fallbrook",
  "Fillmore",
  "Firebaugh",
  "Fish Camp",
  "Folsom",
  "Fontana",
  "Foothill Ranch",
  "Fort Bragg",
  "Fortuna",
  "Foster City",
  "Fountain Valley",
  "Fowler",
  "Fremont",
  "Fresno",
  "Fullerton",
]

def sigmoid(a):
  return 1. / (1 + math.exp(-a))

def bernoulli(a):
  assert 0 <= a <= 1
  return random.random() < a

class Classifier:
  def __init__(self, dim=3, seed=0, lr=0.005):
    if seed:
      random.seed(seed)
    self.params = [0.] * (dim)
    self.dim = dim
    self.lr = lr
    self.num_times_loss = 0

  def transform_data_to_numeric(self, data):
    # name, age, data, city
    return 0, data[1]-(50 - 15)/2, STATES[data[2]], 0

  def predict(self, data):
    assert len(data) == self.dim
    data = self.transform_data_to_numeric(data)
    return sigmoid(sum(self.params[i] * data[i] for i in range(self.dim)))

  def predict_batch(self, data):
    out = []
    for tup in data:
      out.append(self.predict(tup))
    return out

  def learn(self, data, label):
    assert len(data) == self.dim
    data_transformed = self.transform_data_to_numeric(data)
    pred = self.predict(data)
    for i in range(self.dim):
      # print(label, pred, label - pred, data_transformed[i])
      self.params[i] += self.lr * (label - pred) * data_transformed[i]
    # print(self.params)

  def calculate_acc(self, data_batch, label_batch):
    assert len(data_batch[0]) == self.dim

    total = len(data_batch)
    correct = 0
    for pred, label in zip(self.predict_batch(data_batch), label_batch):
      correct += ((pred > 0.5) == label)

    return correct/total

    #self.num_times_loss += 1
    #return 1/(self.num_times_loss * 1.236)   # fake
# rule: theta_j += \alpha(y^{(i)} - h_\theta(x^{(i)})) x^{(i)}_j)

class Generator:
  def __init__(self):
    pass

  def generate(self, fname, num_examples=100):
    with open(fname, 'w') as fout:
      for i in range(num_examples):
        name = random.choice(FIRST_NAMES) + ' ' + random.choice(LAST_NAMES)
        age = random.randint(*AGE_RANGE)
        avg_age = (AGE_RANGE[1] - AGE_RANGE[0]) / 2 + AGE_RANGE[0]
        city = random.choice(CITIES)
        state, state_score = random.choice(list(STATES.items()))
        # print((age - avg_age) * 0.1, state_score*0.5)
        score = sigmoid((age - avg_age) * 0.2 + state_score * -0.2)
        # print(score)
        result = "smoking" if bernoulli(score) else "non-smoking"
        fout.write('\t'.join([name, str(age), state, city, result]) + '\n')

def __main__():
  g = Generator()
  g.generate("train_data.txt", 500)
  g.generate("dev_data.txt", 100)

if __name__ == "__main__":
  __main__()
