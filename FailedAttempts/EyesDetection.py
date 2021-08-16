# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot

class EyesDataset(Dataset):
	# load the dataset definitions


	# function to extract bounding boxes from an annotation file
	def extract_boxes(self,filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		coords = []
		# boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coords = [xmin, ymin, xmax, ymax]
			# boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return coords, width, height

	def load_dataset(self, dataset_dir):
		# define one class
		self.add_class("dataset", 1, "eyes")
		# define data locations
		images_dir_open = dataset_dir + '/OpenFace/'
		annotations_dir_open = dataset_dir + '/OpenFaceAnnotations/'
		images_dir_closed = dataset_dir + '/ClosedFace/'
		annotations_dir_closed = dataset_dir + '/ClosedFaceAnnotations/'
		# find all images
		id = 0
		for filename in listdir(annotations_dir_closed):
			# print(filename)
			img = filename.strip('.xml')
			img_path = images_dir_closed + img
			ann_path = annotations_dir_closed + filename
			# add to dataset
			self.add_image('dataset', image_id=str(id), path=img_path, annotation=ann_path)
			id += 1
		print(id)
		for filename in listdir(annotations_dir_open):
			img = filename.strip('.xml')
			img_path = images_dir_open + img
			ann_path = annotations_dir_open + filename
			# add to dataset
			self.add_image('dataset', image_id=str(id), path=img_path, annotation=ann_path)
			id += 1

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		print(path)
		# load XML
		box, w, h = self.extract_boxes(path)
		print(box)
		# create one array for all masks, each on a different channel
		mask = zeros([h, w], dtype='uint8')
		# create masks
		class_ids = list()
		print(box)
		row_s, row_e = box[1], box[3]
		col_s, col_e = box[0], box[2]
		mask[row_s:row_e, col_s:col_e] = 1
		class_ids.append(self.class_names.index('eyes'))
		return mask, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']


	
# train set
dataset = EyesDataset()
dataset.load_dataset('dataset_B_FacialImages')
dataset.prepare()
print('Dataset: %d' % len(dataset.image_ids))

# load an image
image_id = 1800
image = dataset.load_image(image_id)
print(image.shape)
# load image mask
mask, class_ids = dataset.load_mask(image_id)
print(mask.shape)
# for i in mask:
# 	print(i)
# plot image
pyplot.imshow(image)
# plot mask
pyplot.imshow(mask, cmap='gray', alpha=0.5)
pyplot.show()