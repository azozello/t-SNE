from time import time

import core.engine.data as data
import core.engine.tsne as tsne
import matplotlib.pyplot as plt


time00 = time()
time0 = time()
print("Loading data...")

# Dataset of 28x28 images
filename_train_images = "files/train-images.idx3-ubyte"
filename_train_labels = "files/train-labels.idx1-ubyte"
images_train, labels_train = data.read_MNIST(filename_train_images, filename_train_labels)

n_samples = 1000
images_train = images_train[:n_samples] / 255.
labels_train = labels_train[:n_samples]
images_train = tsne.get_pca_proj(images_train, 30)

print("Done. Time elapsed {:.2f} s".format(time() - time0))

vis = tsne.TSNE(max_iter=1000)
# vis = bh_tsne.BH_TSNE(max_iter=500, bh_threshold=0.5)
vis.fit(images_train, animate=False, labels=labels_train, anim_file="tsne_movie.mp4")

print("Total time: {:.2f} s".format(time()-time00))

fig, ax = plt.subplots()
vis.plot_embedding2D(labels_train, ax)
# plt.savefig('files/samples_' + str(n_samples) + '.png')
plt.show()
