# coco2voc
A tool for converting COCO style annotations to PASCAL VOC style segmentations

Requires pycocotools (see https://github.com/cocodataset/cocoapi). Also, the method does not download the COCO images but instead assumes they exist locally.

Use this to convert the COCO style JSON annotation files to PASCAL VOC style instance and class segmentations in a PNG format. This can be useful when some preprocessing (cropping, rotating, etc.) is required, where it is more convenient to have the labels as images as well.

Class segmentations are an 8-bit PNG images with each pixel value corresponding to the class id of the object in the pixel (https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/). This results in greyscale images that are not visually convenient, but makes more sense when used in training. It is recommended to load the images with PIL so that the values are not normalized (see example code).

In addition to the class and instance segmentations, this also creates an 'ID segmentation' which is a 1-D numpy array in the dimensions of the original image, where the [i, j] cell contains the id of the object at the [i, j] pixel of the image. This can be used to get other information that is not given by the class and instance segmentations (such as bounding boxes etc.).

The ID segmentation can be optionally compressed to an '.npz' file (this is default behavior). These arrays are pretty sparse so the compression is highly effective, but it requires some attention when loading the arrays from file (see example code).

Note that converting the entire dataset can take up to a few hours depending on your machine.
