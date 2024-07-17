#!/bin/python3.10
import ReadData as rd
import perceptron as per

X_train = rd.ReadX('datasets/activity/activity_X_train.txt')
y_train = rd.ReadY('datasets/activity/activity_y_train.txt')
X_test = rd.ReadX('datasets/activity/activity_X_test.txt')
y_test = rd.ReadY('datasets/activity/activity_y_test.txt')
y_train -= 1
y_test -= 1

max_pass = 500
classifiers = per.perceptron_one_vs_all(X_train, y_train, max_pass)

y_train_pred = per.predict_one_vs_all(X_train, classifiers)
y_test_pred = per.predict_one_vs_all(X_test, classifiers)

train_error = per.compute_error(y_train, y_train_pred)
test_error = per.compute_error(y_test, y_test_pred)

print(f'Training Error: {train_error * 100:.2f}%')
print(f'Test Error: {test_error * 100:.2f}%')
