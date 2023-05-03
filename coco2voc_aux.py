import numpy as np
from pycocotools import mask as maskUtils


def annotations_to_seg(annotations, coco_instance):
    """
    converts COCO-format annotations of a given image to a PASCAL-VOC segmentation style label
     !!!No guarantees where segmentations overlap - might lead to loss of objects!!!
    :param annotations: COCO annotations as returned by 'coco.loadAnns'
    :param coco_instance: an instance of the COCO class from pycocotools
    :return: three 2D numpy arrays where the value of each pixel is the class id, instance number, and instance id.
    """
    image_details = coco_instance.loadImgs(annotations[0]['image_id'])[0]

    h = image_details['height']
    w = image_details['width']

    class_seg = np.zeros((h, w))
    instance_seg = np.zeros((h, w))
    id_seg = np.zeros((h, w))
    masks, annotations = annotations_to_mask(annotations, h, w)

    for i, mask in enumerate(masks):
        class_seg = np.where(class_seg > 0, class_seg, mask * annotations[i]['category_id'])
        instance_seg = np.where(instance_seg > 0, instance_seg, mask*(i+1))
        id_seg = np.where(id_seg > 0, id_seg, mask * annotations[i]['id'])

    return class_seg, instance_seg, id_seg.astype(np.int64)


def annotation_to_rle(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        rle = maskUtils.frPyObjects(segm, h, w)  # Uncompressed RLE
    else:
        rle = ann['segmentation']  # RLE
    return rle


def annotations_to_mask(annotations, h, w):
    """
    Convert annotations which can be polygons, uncompressed RLE, or RLE to binary masks.
    :return: a list of binary masks (each a numpy 2D array) of all the annotations in anns
    """
    masks = []
    # Smaller items first, so they won't be covered by overlapping segmentations
    annotations = sorted(annotations, key=lambda x: x['area'])
    for ann in annotations:
        rle = annotation_to_rle(ann, h, w)
        m = maskUtils.decode(rle)
        masks.append(m)
    return masks, annotations
