# coco2voc
A tool for converting COCO style annotations to PASCAL VOC style segmentations

Use this to convert the COCO style JSON annotation files to PASCAL VOC style instance and class segmentations in a PNG format. This can be useful when some preprocessing (cropping, rotating, etc.) is required, where it is more convenient to have the labels as images as well.

In addition to the class and instance segmentations, this also creates an 'ID segmentation' which is a 1-D numpy array in the dimensions of the original image, where the [i, j] cell contains the id of the object at the [i, j] pixel of the image. This can be used to get other information that is not given by the class and instance segmentations (such as bounding boxes etc.).

The ID segmentation can be optionally compressed to an '.npz' file (this is default behavior). These arrays are pretty sparse so the compression is highly effective, but it requires some attention when loading the arrays from file (see example code)
