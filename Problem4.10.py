import cPickle
import PIL
import scipy.misc
import numpy
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import mean_squared_error
import plotly.plotly as py
import plotly.graph_objs as go

# Converts the raw row vectors to 32 X 32 image arrays
def row_to_img(row_vector):
	my_img = numpy.array([ numpy.array([numpy.array([None for z in range(3)]) for y in range(32)]) for x in range(32)])
	
	for row in range(32):
		for col in range(32):
			
			red = row_vector[row*32 + col]
			green = row_vector[row*32 + col + 1024]
			blue = row_vector[row*32 + col + 2048]

			pixel = numpy.array([red, green, blue])
			my_img[row][col] = pixel
	
	return my_img

def write_img_to_file(img, filename):
	scipy.misc.imsave(filename, img)

def get_class_names():
	class_names = [line.rstrip('\n') for line in open('dataset/names.txt')]
	return class_names

# DONT NEED THIS ANYMORE
def compute_mean_imgs():
	# Files containing training data
	train_files = ["dataset/data_batch_1", "dataset/data_batch_2", "dataset/data_batch_3", "dataset/data_batch_4", "dataset/data_batch_5"]

	# Store the mean image for each class
	mean_images = numpy.array([ numpy.array([0 for y in range(3072) ]) for x in range(10)])

	# Store the number of images in each class
	class_count = [ 0 for x in range(10)]

	for curr_file in train_files:
		# Open the current file
		f = open(curr_file)

		# Unpack the dataset
		my_dict = cPickle.load(f)

		# Separate data and labels
		train_data = my_dict['data']
		train_labels = my_dict['labels']

		for i in range(len(train_labels)):
			# increase the count
			class_count[train_labels[i] - 1] = class_count[train_labels[i] - 1] + 1

			# Modify the current image vector
			mean_images[train_labels[i] - 1] = mean_images[train_labels[i] - 1] + train_data[i]

		# Compute the mean
		for i in range(10):
			mean_images[i] = mean_images[i] / class_count[i]

		# Close file
		f.close()

	print class_count
	return mean_images

def get_principal_components(num):
	# Files containing training data
	train_files = ["dataset/data_batch_1", "dataset/data_batch_2", "dataset/data_batch_3", "dataset/data_batch_4", "dataset/data_batch_5"]

	images = numpy.array([numpy.array([0 for y in range(3072)]) for x in range(5000)])

	pca_components = numpy.array([None for x in range(10)])

	# Do PCA separately for each category
	for label in range(10):

		# Store the number of images in each class
		count = 0

		for curr_file in train_files:
			# Open the current file
			f = open(curr_file)

			# Unpack the dataset
			my_dict = cPickle.load(f)

			# Separate data and labels
			train_data = my_dict['data']
			train_labels = my_dict['labels']

			for i in range(len(train_labels)):
				# Matches the current label
				if label == (train_labels[i]) and (count < 5000):
					images[count] = train_data[i]

					count = count + 1

			f.close()

		# Memory Efficient PCA
		ipca = IncrementalPCA(n_components = num, batch_size = 200)
		x_ipca = ipca.fit(images)
		pca_components[label] = x_ipca

	return pca_components

def get_test_error(pca_components):
	error = numpy.array([0 for x in range(10)])
	count = numpy.array([0 for x in range(10)])

	f = open('dataset/test_batch')

	# Unpack the dataset
	my_dict = cPickle.load(f)

	# Separate data and labels
	test_data = numpy.array(my_dict['data'])
	test_labels = my_dict['labels']

	for i in range(len(test_labels)):

		# class = test_labels[i]
		test_data_transform = pca_components[test_labels[i]].transform(numpy.array(test_data[i]))
		count[test_labels[i]] = count[test_labels[i]] + 1

		# convert vector back into original space
		inverse = pca_components[test_labels[i]].inverse_transform(test_data_transform)

		# find mean squared error
		error[test_labels[i]] = error[test_labels[i]] + mean_squared_error(test_data[i], inverse[0])

	f.close()

	# Error averaged over the number of images
	error = [error[i]/count[i] for i in range(10)]

	return error

# Returns a 10 x 10 matrix where (i, j) corresponds to the distance between mean images of class i and j
def get_mean_distances(mean_images):
	matrix = numpy.array([numpy.array([0 for y in range(10)]) for x in range(10)])

	for i in range(10):
		for j in range(10):
			matrix[i][j] = mean_squared_error(mean_images[i], mean_images[j])

	return matrix


def main():
	# Get the class names
	class_names = get_class_names()

	# Get pca objects for each category
	pca_components = get_principal_components(20)

	# Convert Mean Components to Images
	for i in range(10):
		img = row_to_img(pca_components[i].mean_)
		write_img_to_file(img, class_names[i] + ".png")

	# plot error on test images
	mean_errors =  get_test_error(pca_components)
	data = [go.Bar(x = class_names, y = mean_errors)]
	py.plot(data, filename = "PCA Test Errors")

	# Compute Distances
	#mean_dist = get_mean_distances([pca_components[i].mean_ for i in range(10)])

	#print mean_dist

main()