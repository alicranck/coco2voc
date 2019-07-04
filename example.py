from coco2voc import *
from PIL import Image

def on_press(event):
    """
    Keyboard interaction ,key `a` for next image, key `d` for previous image, key `t` segmentation toggle
    :param event: :class:`~matplotlib.backend_bases.KeyEvent`, keyboard event
    :return: None
    """
    global i, l, frames, segs, fplot, splot, fig, ax, s_toggle, id_list, figsizes

    if event.key == 'd':
        i = (i+1) % l
        s_toggle = True
        splot.set_alpha(0.4)
    elif event.key == 'a':
        i = (i-1) % l
        s_toggle = True
        splot.set_alpha(0.4)
    elif event.key == 't':
        # show or hide segmentation
        s_toggle = not s_toggle
        splot.set_alpha(0.4) if s_toggle else splot.set_alpha(0)

    fplot.set_data(frames[i])
    splot.set_data(segs[i])

    fig.set_size_inches(figsizes[i], forward=True)
    fig.canvas.draw()
    ax.set_title(id_list[i])

    pass


if __name__ == '__main__':
    # !!Change paths to your local machine!!
    annotations_file = '/home/dl/1TB-Volumn/MSCOCO2017/annotations/instances_train2017.json'
    labels_target_folder = '/home/dl/PycharmProjects/coco2voc-master/output'
    data_folder = '/home/dl/1TB-Volumn/MSCOCO2017/train2017'


    # Convert n=25 annotations
    coco2voc(annotations_file, labels_target_folder, n=25, compress=True)

    # Load an image with it's id segmentation and show
    coco = COCO(annotations_file)
    path = os.path.join(labels_target_folder, 'images_ids.txt')

    # Read ids of images whose annotations have been converted from specified file
    with open(path) as f:
        id_list = [line.split()[0] for line in f]

    i = 0
    l = len(id_list)
    frames = []
    segs =[]
    figsizes = []
    s_toggle = True
    dpi = 100

    for id in id_list:
        # Get the image's file name and load image from data folder
        img_ann = coco.loadImgs(int(id))

        file_name = img_ann[0]['file_name']
        im_data = plt.imread(os.path.join(data_folder, file_name))
        height, width, depth = im_data.shape
        frames.append(im_data)

        size = width/float(dpi), height/float(dpi)
        figsizes.append(size)

        # Load segmentation - note that the loaded '.npz' file is a dictionary, and the data is at key 'arr_0'
        id_seg = np.load(os.path.join(labels_target_folder, 'id_labels', id +'.npz'))
        segs.append(id_seg['arr_0'])
        
        # Example for loading class or instance segmentations
        instance_filename = os.path.join(labels_target_folder, 'instance_labels', id +'.png')
        class_filename = os.path.join(labels_target_folder, 'class_labels', id +'.png')
        instance_seg = np.array(Image.open(instance_filename))
        class_seg = np.array(Image.open(class_filename))


    # Show image with segmentations
    fig, ax = plt.subplots(figsize=figsizes[0], dpi=dpi)
    fig.canvas.mpl_connect('key_press_event', on_press)

    fplot = ax.imshow(frames[i%l])
    splot = ax.imshow(segs[i%l], alpha=0.4)

    ax.set_aspect(aspect='auto')# must after imshow

    plt.tight_layout()
    plt.axis('off')
    plt.show()
