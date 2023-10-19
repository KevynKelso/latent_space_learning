# Taken from this paper: https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
from threading import Thread
from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import savez_compressed
from os import listdir


def load_image(filename):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    return pixels

def load_faces(directory, n_faces):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # load the image
        pixels = load_image(directory + filename)
        # store
        faces.append(pixels)
        # stop once we have enough
        if len(faces) >= n_faces:
            break
    return asarray(faces)

def plot_faces(faces, n):
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(faces[i])
    pyplot.show()

def extract_face(model, pixels, required_size=(80, 80)):
	# detect face in the image
	faces = model.detect_faces(pixels)
	# skip cases where we could not detect a face
	if len(faces) == 0:
		return None
	# extract details of the face
	x1, y1, width, height = faces[0]['box']
	# force detected pixel values to be positive (bug fix)
	x1, y1 = abs(x1), abs(y1)
	# convert into coordinates
	x2, y2 = x1 + width, y1 + height
	# retrieve face pixels
	face_pixels = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face_pixels)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def run_face(image, model, faces):
	# load the image
	pixels = load_image(image)
	# get face
	face = extract_face(model, pixels)
	if face is None:
		return
	# store
	faces.append(face)

def load_faces_mtcnn_sequential(directory):
	# prepare model
	model = MTCNN()
	faces = list()
	max_faces = 5000
	count = 0
	checkpoint_count = 0
	# enumerate files
	for filename in listdir(directory):
		count += 1
		run_face(directory + filename, model, faces)
		if count >= max_faces:
			checkpoint_count += 1
			count = 0
			savez_compressed(f'img_align_celeba_{checkpoint_count}.npz', asarray(faces))
			print(f"checkpoint {checkpoint_count}")
			faces = list()

def load_faces_mtcnn_threaded(directory):
	# prepare model
	model = MTCNN()
	faces = list()
	num_threads = 250
	threads = []
	count = 0
	# enumerate files
	for filename in listdir(directory):
		if len(threads) < num_threads:
			threads.append(Thread(target=run_face, args=(directory + filename, model, faces)))
		else:
			[t.start() for t in threads]
			[t.join() for t in threads]
			savez_compressed(f'img_align_celeba_{count}.npz', asarray(faces))
			print(f"checkpoint {count}")
			count += 1
			faces = list()
			threads = []
			

def celeba_preproc():
    directory = 'img_align_celeba/'
    # faces = load_faces(directory, 25)
    # print('Loaded: ', faces.shape)
    # plot_faces(faces, 5)
    load_faces_mtcnn_sequential(directory)


if __name__ == '__main__':
    celeba_preproc()
