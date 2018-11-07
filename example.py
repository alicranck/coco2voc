from coco2voc import *


# !!Change paths to your local machine!!
annotations_file = 'E:\Almog\DLProjectData\coco\\annotations_trainval2017\\annotations\\instances_val2017.json'
labels_target_folder = 'E:\Almog\DLProjectData\coco\\example_labels\\'
data_folder = 'E:\Almog\DLProjectData\coco\\val2017\\'

# Convert n=25 annotations
coco2voc(annotations_file, labels_target_folder, n=25, compress=True)

# Load an image with it's id segmentation and show
coco = COCO(annotations_file)

# Read ids of images whose annotations have been converted from specified file
id_list = open(labels_target_folder + 'images_ids.txt')
line = id_list.readline()
id = line.split()[0]

# Get the image's file name and load image from data folder
img_ann = coco.loadImgs(int(id))
file_name = img_ann[0]['file_name']
img = plt.imread(data_folder + file_name)

# Load segmentation - note that the loaded '.npz' file is a dictionary, and the data is at key 'arr_0'
id_seg = np.load(labels_target_folder+'id_labels\\'+id+'.npz')
seg = id_seg['arr_0']

# Show image with segmentations
plt.imshow(img)
plt.imshow(id_seg['arr_0'], alpha=0.4)
plt.show()

