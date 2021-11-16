### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split
from Configure import model_configs, training_configs

from datetime import datetime
dt=datetime.now().strftime('%m_%d_%Hh_%Mm')
parser = argparse.ArgumentParser()
# parser.add_argument("mode", help="train, test or predict")
# parser.add_argument("data_dir", help="path to the data")
# parser.add_argument("--save_dir", help="path to save the results")

parser.add_argument("mode", help="train")
parser.add_argument("data_dir", help="G:\Tamu\Semester 1\Deep Learning\Project\CSCE636-project-2021Fall\data")
parser.add_argument("--save_dir", help="G:\Tamu\Semester 1\Deep Learning\Project\CSCE636-project-2021Fall\output\result_dt")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_test, y_test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test)

	elif args.mode == 'predict':
		# Predicting and storing results on private testing dataset 
		x_test = load_testing_images(args.data_dir)
		predictions = model.predict_prob(x_test)
		np.save(args.result_dir, predictions)
		

### END CODE HERE

