from coco2voc import *


def on_press(event):
    global i, l, frames, segs, fplot, splot, fig, s_toggle

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

    fig.canvas.draw()
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
    s_toggle = True

    for id in id_list:
        # Get the image's file name and load image from data folder
        img_ann = coco.loadImgs(int(id))
        file_name = img_ann[0]['file_name']
        frames.append(plt.imread(os.path.join(data_folder, file_name)))

        # Load segmentation - note that the loaded '.npz' file is a dictionary, and the data is at key 'arr_0'
        id_seg = np.load(os.path.join(labels_target_folder, 'id_labels', id+'.npz'))
        segs.append(id_seg['arr_0'])

    # Show image with segmentations
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)
    print(l)
    fplot = ax.imshow(frames[i % l], aspect=1)
    splot = ax.imshow(segs[i%l], alpha=0.4, aspect=1)
    plt.show()

