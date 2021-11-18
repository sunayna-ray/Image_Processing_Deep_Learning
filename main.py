### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, load_private_testing_images
from Configure import model_configs, training_configs

from datetime import datetime
dt=datetime.now().strftime('%m_%d_%Hh_%Mm')
# parser = argparse.ArgumentParser()
# parser.add_argument("mode", help="train, test or predict")
# parser.add_argument("data_dir", help="path to the data")
# parser.add_argument("--save_dir", help="path to save the results")
# args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)

	mode = 'predict'
	data_dir = "G:\Tamu\Semester 1\Deep Learning\Project\CSCE636-project-2021Fall\starter_code\data"
	result_dir = "G:\Tamu\Semester 1\Deep Learning\Project\CSCE636-project-2021Fall\output\result_dt"
	# if (args.mode!=''): mode= args.mode
	# if (args.data_dir!=''): data_dir= args.data_dir
	# if (args.result_dir!=''): result_dir= args.result_dir

	if mode == 'train':
		train_dataset_loaded, valid_dataset_loaded, _ = load_data(data_dir)

		# model.train(x_train, y_train, training_configs, x_valid, y_valid)
		# model.evaluate(x_test, y_test)

	elif mode == 'test':
		# Testing on public testing dataset
		_, _, cifar_test_dataset_loaded = load_data(data_dir)
		# model.evaluate(x_test, y_test)

	elif mode == 'predict':
		# Predicting and storing results on private testing dataset 
		x_test = load_private_testing_images(data_dir)
		# predictions = model.predict_prob(x_test)
		# np.save(result_dir, predictions)
		

### END CODE HERE

