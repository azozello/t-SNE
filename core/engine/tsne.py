import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from time import time


class TSNE(object):
    """
    Implements basic t-Distributed Stochastic Neighbor Embedding.
    """

    def __init__(self, n_components=2, max_iter=1000, learning_rate=500.,
        momentum=0.5, momentum_final=0.8, n_momentum=250, early_exaggeration=4.,
        n_early_exag=100, init_method='pca', perplexity=30, perplex_tol=1e-4,
        perplex_evals_max=50, min_grad_norm2=1e-14, cost_min_since_max=30):
        """
        Set initial parameters for t-SNE.
        Input:
        - n_components: dimensionality of visualization
        - max_iter: max number of iterations
        - learning_rate: for gradient descent
        - momentum: for gradient descent
        - momentum_final: momentum after n_momentum iterations
        - n_momentum: change momentum to final after this
        - early exaggeration: factor for affinities in high dimension
        - n_early_exag: apply the above until this
        - init_method: initialization method 'pca' or 'rnorm'
        - perplexity: entropy requirement for bandwidth
        - perplex_tol: tolerance for error in evaluated entropy
        - perplex_evals_max: max number of tries for ok bandwidth
        - min_grad_norm2: abort if squared norm of gradient below this
        - cost_min_since_max: abort if no progress these iterations
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_final = momentum_final
        self.n_momentum = n_momentum
        self.early_exaggeration = early_exaggeration
        self.n_early_exag = n_early_exag
        self.init_method = init_method
        self.perplexity = perplexity
        self.perplex_tol = perplex_tol
        self.perplex_evals_max = perplex_evals_max
        self.min_grad_norm2 = min_grad_norm2
        self.cost_min_since_max = cost_min_since_max

    def fit(self, data, animate=False, labels=None, anim_file="tsne_movie.mp4"):
        """
        Apply the t-SNE to input data.
        If animate=True, labels must be provided.
        """
        self.data = data  # high-dimensional input data
        self.n_samples = data.shape[0]
        # self.n_dim = data.shape[1]

        # evaluate p_ij
        time0 = time()
        print("Calculating affinities in high dimension...")
        self._set_affin_hd()
        print("Done. Time elapsed {:.2f} s".format(time() - time0))

        # apply early exaggeration
        self.affin_hd *= self.early_exaggeration

        # initialize visualization
        time0 = time()
        if self.init_method == 'pca':
            print("Calculating PCA as initial guess...")
            self.coord = get_pca_proj(data, self.n_components)

        elif self.init_method == 'rnorm':
            print("Sampling initial distribution...")
            self.coord = np.random.randn(self.n_samples, self.n_components) * 1e-4

        print("Done. Time elapsed {:.2f} s".format(time() - time0))

        # animate
        if animate and labels is not None:
            print("Recording animation.")
            writer = anim.writers['ffmpeg'](fps=10)
            fig, ax = plt.subplots()
            markers = self.plot_embedding2D(labels, ax)

            with writer.saving(fig, anim_file, 160):
                self._iterate(markers=markers, writer=writer)

        # don't animate
        else:
            self._iterate()

    def _set_affin_hd(self):
        """
        Calculate pairwise affinities in high dimensional data.
        """
        affin = np.zeros((self.n_samples, self.n_samples))

        # calculate conditional probabilities
        for ii in range(self.n_samples):
            # all squared distances from ii
            dist2 = np.sum((self.data - self.data[ii])**2., axis=1)

            affin[ii, :] = self._affin_bin_search_sigma(dist2)

            # TODO optimize
            # dist2 = np.sum( (self.data[ii+1:] - self.data[ii])**2., axis=1 )
            # affin[ii, ii+1:] = self._affin_bin_search_sigma(dist2)
            # affin[ii, :ii+1] = affin[:ii+1, ii]

        affin.flat[::self.n_samples+1] = 1.e-12  # set p_ii ~= 0
        affin = np.where(affin < 1.e-12, 1.e-12, affin)
        affin = (affin + affin.T) / (2. * self.n_samples)  # make symmetric

        self.affin_hd = affin

    def _affin_bin_search_sigma(self, dist2):
        """
        Binary search for sigma to get to desired perplexity.
        Return affinity.
        """
        log_perplexity = np.log2(self.perplexity)
        s_min = 0.
        s_max = 1.e15
        error = self.perplex_tol + 1.  # just set big enough
        evals = 0
        sigma22 = 1.e5  # initial guess for 2*sigma^2

        while abs(error) > self.perplex_tol and evals < self.perplex_evals_max:

            # calculate affinities for current sigma
            denom = np.sum( np.exp( -dist2[dist2>0.] / sigma22 ) )
            # if dist2.shape[0] == self.n_samples:
            #     denom -= 1.  # remove contribution from self
            affin = np.exp( -dist2 / sigma22 ) / denom

            # Shannon entropy = log2(perplexity)
            shannon = -np.sum(np.where(affin > 0., affin * np.log2(affin.clip(min=1e-12)), 0.))
            error = shannon - log_perplexity

            # P and Shannon entropy increase as sigma increases
            if error > 0:  # -> sigma too large
                s_max = sigma22
                sigma22 = (sigma22 + s_min) / 2.
            else:  # -> sigma too small
                s_min = sigma22
                sigma22 = (sigma22 + s_max) / 2.

            evals += 1

        return affin

    def _set_gradient(self):
        """
        Calculate pairwise affinities in 2D data using
        Student's t-distribution with 1 degree of freedom.

        Then calculate the gradient with respect to each 2D point.
        """

        # all pairwise distances
        dist2 = np.sum( (self.coord - self.coord[:, np.newaxis])**2., axis=2)

        student = 1. / (1. + dist2)
        student.flat[::self.n_samples+1] = 1.e-12  # set q_ii ~= 0
        student = np.where(student < 1.e-12, 1.e-12, student)
        self.affin_ld = student / np.sum( student )

        self.gradient = 4. * np.sum(
                ((self.affin_hd - self.affin_ld) * student)[:, :, np.newaxis]
                        * (self.coord - self.coord[:, np.newaxis])
                                    , axis=1)

    def _iterate(self, markers=None, writer=None):
        """
        Iterate using gradient descent.
        """
        print("Iterating t-SNE...")

        coord_diff = np.zeros_like(self.coord)
        stepsize = np.ones_like(self.coord) * self.learning_rate

        print_period = 10
        ii = 0
        grad_norm2 = 1.

        if self.cost_min_since_max > 0:
            cost_min = 1e99
            cost_min_since = 0
            costP = np.sum( self.affin_hd * np.log(self.affin_hd.clip(min=1e-12)) )

        time0 = time()
        while ii < self.max_iter and grad_norm2 > self.min_grad_norm2:
            if ii > 0 and ii % print_period == 0:
                print( "{} iterations done. Time elapsed for last {}: "
                       "{:.2f} s. Gradient norm {:f}."
                       .format(ii, print_period, time() - time0, np.sqrt(grad_norm2)) )
                time0 = time()

            if ii == self.n_early_exag:
                self.affin_hd /= self.early_exaggeration  # cease "early exaggeration"

            if ii == self.n_momentum:
                self.momentum = self.momentum_final

            self._set_gradient()

            if ii > self.n_early_exag and self.cost_min_since_max > 0:
                # abort if no progress for a while
                cost = costP - np.sum( self.affin_hd * np.log(self.affin_ld) )
                if cost < cost_min:
                    cost_min = cost
                    cost_min_since = 0
                else:
                    cost_min_since += 1
                    if cost_min_since > self.cost_min_since_max:
                        print( "No progress in {} iterations. Aborting."
                               .format(cost_min_since) )
                        break

            # Decrease stepsize if previous step and current gradient in the same direction.
            # Otherwise increase stepsize (note the negative definition of gradient here).
            stepsize = np.where( (self.gradient > 0) == (coord_diff > 0),
                                np.maximum(stepsize * 0.8, 0.01 * self.learning_rate),
                                stepsize + 0.2 * self.learning_rate)

            coord_diff = self.learning_rate * self.gradient \
                         + self.momentum * coord_diff
            self.coord += coord_diff

            # update animation
            if writer:
                print("Animating iteration {}".format(ii))
                data = normalize(self.coord)
                for jj, text_artist in enumerate(markers):  # SLOW!
                    text_artist.set_x(data[jj, 0])
                    text_artist.set_y(data[jj, 1])
                writer.grab_frame()

            grad_norm2 = np.sum (self.gradient**2.)
            ii += 1

    def plot_embedding2D(self, labels, ax):
        """
        Plot the 2D data with labels.
        """
        n_class = 1. * len(np.unique(labels))

        markers = []  # list to hold text artists
        data = normalize(self.coord)

        # plot a number colored according to label
        for ii in range(self.n_samples):
            text_artist = ax.text(data[ii, 0], data[ii, 1], str(labels[ii]),
                                  color=plt.cm.Set1(labels[ii] / n_class),
                                  fontdict={'weight': 'bold', 'size': 12})
            markers.append(text_artist)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        return markers


def normalize(data):
    """
    Return data normalized to [0,1].
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data = (data - data_min) / (data_max - data_min)
    return data


def get_pca_proj(data, n_components):
    """
    Return the data projected on its first principal components.
    """
    # contruct covariance matrix
    covmat = np.cov(data, rowvar=False)
    # get eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(covmat)
    # sort eigenvectors (ascending) and pick some of the highest
    inds = np.argsort(eigval)[-n_components:]
    # project data on first components
    proj = data.dot(eigvec[:, inds])

    return proj
