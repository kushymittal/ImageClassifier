import cPickle
import PIL
import scipy.misc
import numpy
from sklearn.decomposition import PCA


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

def row_to_img_32_cross_96(row_vector):
	my_img = numpy.array([numpy.array([None for y in range(96)]) for x in range(32)])

	for row in range(32):
		for col in range(96):

			col_in_img = col % 32

			if col < 32:
				my_img[row][col] = row_vector[row*32 + col_in_img]
			elif col < 64:
				my_img[row][col] = row_vector[row*32 + col_in_img + 1024]
			else:
				my_img[row][col] = row_vector[row*32 + col_in_img + 2048]
	return my_img


def write_img_to_file(img, filename):
	pass
	#scipy.misc.imsave(filename, img)

def get_class_names():
	class_names = [line.rstrip('\n') for line in open('dataset/names.txt')]

	return class_names

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

	return mean_images

def main():
	means = compute_mean_imgs()
	class_names = get_class_names()

	for img_vectors in range(len(means)):
		img = row_to_img_32_cross_96(means[img_vectors])

		write_img_to_file(img, class_names[img_vectors] + ".png")
		print img.shape
		pca = PCA(n_components = 20)
		pca.fit(img)
		print(pca.explained_variance_ratio_)
		break

main()