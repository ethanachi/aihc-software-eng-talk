import ml_library as ml

def report_results(classifier, dev):
  dev_predictions = classifier.predict_batch(dev.get_data_batch())
  num_printed = 0
  for data, label, prediction in zip(dev.get_data_batch(), dev.get_label_batch(), dev_predictions):
    print(data[0] + '\t' + str(data[1]) + '\t' + data[2] + '\t' + data[3] +
          '\t' + str(label) + '\t' + str(prediction), ((prediction > 0.5) == label))
    num_printed += 1
    if num_printed == 5: break

def report_results_c(args, classifier, dev):
  dev_predictions = classifier.predict_batch(dev.get_data_batch())
  num_printed = 0
  for data, label, prediction in zip(dev.get_data_batch(), dev.get_label_batch(), dev_predictions):
    print(data[0] + '\t' + str(data[1]) + '\t' + data[2] + '\t' + data[3] +
          '\t' + str(label) + '\t' + str(prediction), ((prediction > 0.5) == label))
    num_printed += 1
    if num_printed == args['num_to_print']: break
