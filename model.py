# each row is the steering angle, throttle, brake, and speed of your car
# measurements are the steering angle
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from sklearn.utils import shuffle

correction = 0.25
total_path = '/home/carnd/CarND-Behavioral-Cloning-P3/train_data'

lines = []
with open('/home/carnd/CarND-Behavioral-Cloning-P3/train_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '/home/carnd/CarND-Behavioral-Cloning-P3/train_data/IMG/' + filename
	image = cv2.imread(current_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	images.append(image)
	measurements.append(float(line[3]))
	
	# left camera image
	source_path_l = line[1]
	left_filename = source_path_l.split('/')[-1]
	# right camera image
	source_path_r = line[2]
	right_filename = source_path_r.split('/')[-1]

	left_path = '{}/IMG/'.format(total_path) + left_filename
	image = cv2.imread(left_path)
	images.append(image)
	# adjust the angle
	measurements.append(float(line[3]) + correction)


	right_path = '{}/IMG/'.format(total_path) + right_filename
	image = cv2.imread(right_path)
	images.append(image)
	# flip the image
	# images.append(np.fliplr(image))
	# adjust the angle
	measurements.append(float(line[3]) - correction)
	# measurements.append(-(float(line[3]) - correction))


X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x : (x/255.0) - 0.5, input_shape=(160,320,3)))
# the width of first convolution layer output is 24, number of convolution layers,
# kernal size is 5*5,
# stride is 2,
model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')


