### Assignment 1 ###
# Generic MP Neuron
# Input Vector of length 'm'
# Threshold 't'
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

#### Ques 1 ####

def mp_neuron(t, input_sum):
	""" 
	McCulloch-Pitts Neuron for prediction.
	Parameters:
	t (int) : threshold input by user
	input_sum (int) : sum of the vector input by user
	Returns:
	(int) : Boolean answer which is typecasted to int 
	to get 0 or 1 for False or True, respectively.
	"""
	return int(input_sum>=t)


string_vector = input('Vector: ')
# Input always takes the type string. We split the array over commas 
# and we change data type from unicode to integer of size 16 bits (-32768 to 32767).
integer_vector = np.array(string_vector.split(','), dtype=np.int16)
sum_int = sum(integer_vector)

string_threshold = input('t: ')
integer_threshold = int(string_threshold)

result = mp_neuron(integer_threshold, sum_int)
print("Result: ", result)

#### Ques 2 ####

def accuracy(match_cases, pred_result, act_result): 
	"""
	This function matches the predicted results with actual results and counts 
	the number of cases where both of these match.
	Parameters:
		match_cases (int) : number of cases matched
		pred_result (bool) : result predicted by model
		act_result (bool) : actual result
	Returns:
		match_cases (int) : returns match_cases after updating value
	"""
	if(pred_result==act_result):
		match_cases+=1
	return match_cases

# Stores accuracy of all thresholds 0 to 10 (10 inclusive). 
accuracy_list = []
# Stores list of sums of all inputs from x1 to x10.
sum_list = []
# Stores list of actual results given in the database.
result_list = []

with open("Assignment1.csv") as file:
	csv_reader = csv.reader(file)
	# Removing header of the file
	header = next(csv_reader)
	if header != None: # checking for empty file
		for row in file:
			# converting string to array of integers
			int_row_with_res = np.array(row.split(','), dtype=np.int16)
			# splitting to get values x1 to x10
			int_row = int_row_with_res[:10]
			sum_row = sum(int_row)
			sum_list.append(sum_row)
			# splitting to get y (we could also use pandas 
			# and pandas.DataFrame to get the column)
			int_res = int(int_row_with_res[10:])
			result_list.append(int_res)

# getting accuracy:
for t in range(0, 11): # 10 is inclusive
	match_cases = 0
	for i in range(0, 1000):
		prediction = mp_neuron(t, sum_list[i])
		match_cases = accuracy(match_cases, prediction, result_list[i])
	acc = match_cases/1000
	accuracy_list.append(acc)

print("Accuracy: ", accuracy_list)

#### Ques 3 ####

thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# to change font size of all graphs
plt.rcParams.update({'font.size': 7}) 

# Line Graph
plt.plot(thresholds, accuracy_list, marker='o')
plt.title('Ques 3')
plt.xlabel('Threshold Value')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xlim(0, 10)
# Annotating:
for x, y in zip(thresholds, accuracy_list):
	label = "({0}, {1})".format(x, y)
	plt.annotate(label, # text
				(x, y), # point to label
				textcoords="offset points", # how to position text
				xytext=(0, 10), # distance from text to (x, y)
				ha = 'center') # horizontal alignment
plt.show()

# Grid Chart: for better visualization
Data = {'Threshold Value' : thresholds, 'Accuracy' : accuracy_list}
df = pd.DataFrame(Data, columns = ['Threshold Value', 'Accuracy'])
plt.plot(df['Threshold Value'], df['Accuracy'], color = 'red', marker = 'o')
plt.title('Ques 3')
plt.xlabel('Threshold Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.ylim(0, 1)
plt.xlim(0, 10)
plt.show()

####### Made By: Shanya Singhal #######
####### SID: 17103089 #################
		