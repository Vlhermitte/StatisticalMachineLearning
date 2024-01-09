# Solving image shape using EM algorithm (Valentin Lhermitte)
import numpy as np
import matplotlib.pyplot as plt
import tarfile
from scipy.special import softmax


def get_images(tar_file, test=False):
    # upack the tar file
    tar = tarfile.open(tar_file)
    tar.extractall()
    tar.close()
    # load the images
    if test:
        images = np.load('images0.npy', allow_pickle=True)
    else:
        images = np.load('images.npy', allow_pickle=True)
    return images

def e_step(avg_image, estimated_img, eta_0, eta_1):
    # compute the likelihoods
    likelihoods = (estimated_img * (avg_image * eta_1 - np.log(1 + np.exp(eta_1))) +
                   (1 - estimated_img) * (avg_image * eta_0 - np.log(1 + np.exp(eta_1))))
    # compute the number of 0 and 1 pixels
    n_1 = np.sum(estimated_img)
    n_0 = np.size(estimated_img) - n_1
    return n_0, n_1, likelihoods

def m_step(avg_image, estimated_img, n_0, n_1):
    """
    Computes the new parameters for the model.
    :param avg_image: The average image.
    :param estimated_img: The estimated image.
    :param n_0: The number of 0 pixels.
    :param n_1: The number of 1 pixels.
    :return: The new parameters.
    """
    # compute the new parameters
    etas_0 = np.sum((1 - estimated_img) * avg_image) / n_0
    etas_1 = np.sum(estimated_img * avg_image) / n_1
    # return the parameters
    return etas_0, etas_1

def shape_mle(avg_image, ethas_init):
    estimated_img = np.random.randint(2, size=avg_image.shape)
    # initialise the parameters
    eta_0 = ethas_init[0]
    eta_1 = ethas_init[1]

    # set up a counter and a likelihood
    epsilon = 100
    n_iterations = 100
    i = 0
    # loop over the iterations
    while i < n_iterations and abs(epsilon) > 0.01:
        # E-step
        n_0, n_1, likelihoods = e_step(avg_image, estimated_img, eta_0, eta_1)
        average_pixel = np.sum(likelihoods) / np.size(likelihoods)
        # compute the new image
        estimated_img = np.zeros(avg_image.shape)
        estimated_img[likelihoods >= average_pixel] = 1
        # M-step
        new_eta_0, new_eta_1 = m_step(avg_image, estimated_img, n_0, n_1)
        epsilon = (new_eta_0 + new_eta_1) - (eta_0 + eta_0)
        eta_0 = new_eta_0
        eta_1 = new_eta_1
        # increment the counter
        i += 1
    # return the estimated image and the parameters
    return estimated_img, [eta_0, eta_1]

def Assignment1():
    print("Assignment 1")
    # Generate 2 sets of images with different means and standard deviations
    images = get_images('em_data.tar', test=True)
    avg_image = np.average(images, axis=0)

    # initialise the parameters
    etas = [0, 1]

    estimated_img, etas = shape_mle(avg_image, etas)
    print("Etas : ", etas)

    plt.imshow(estimated_img, cmap='gray')
    plt.title("Estimated shape")
    plt.axis('off')
    plt.show()


def posterior_pose_probs(images, etas, s, pi):
    """
    Computes the array of alpha's for all images.

    :param images: A numpy array of images (x) of shape (m, n), where m is the number of images and n is the number of pixels.
    :param etas: The 2-tuple of etas.
    :param s: The current estimate of the shape.
    :param pi: The current estimate of the pose probabilities (pi_r).
    :return: A numpy array of alpha's of shape (m, r), where r is the number of poses.
    """
    prod0 = np.sum(images * (etas[0] *(1 - s) + etas[1] * s), axis=(1,2)) + np.log(pi[0])
    prod1 = np.sum(images * (etas[0] * (1 - np.rot90(s)) + etas[1] * np.rot90(s)), axis=(1,2)) + np.log(pi[1])
    prod2 = np.sum(images * (etas[0] * (1 - np.rot90(s, k=2)) + etas[1] * np.rot90(np.rot90(s))), axis=(1,2)) + np.log(pi[2])
    prod3 = np.sum(images * (etas[0] * (1 - np.rot90(s, k=-1)) + etas[1] * np.rot90(np.rot90(np.rot90(s)))), axis=(1,2)) + np.log(pi[3])

    alpha = np.stack((prod0, prod1, prod2, prod3), axis=1)
    alpha = softmax(alpha, axis=1)
    return alpha

def assignment2_m_step(images, etas, alpha):
    psi0 = alpha[:, 0][:, np.newaxis, np.newaxis] * images
    psi1 = alpha[:, 1][:, np.newaxis, np.newaxis] * np.rot90(images, k=3, axes=(1,2))
    psi2 = alpha[:, 2][:, np.newaxis, np.newaxis] * np.rot90(images, k=2, axes=(1,2))
    psi3 = alpha[:, 3][:, np.newaxis, np.newaxis] * np.rot90(images, k=1, axes=(1,2))

    psi = np.sum(psi0 + psi1 + psi2 + psi3, axis=0) / images.shape[0]
    s, etas = shape_mle(psi, etas)
    pi = np.mean(alpha, axis=0)
    return s, etas, pi

def Assignment2():
    print("Assignment 2")
    images = get_images('em_data.tar', test=False)
    avg_image = np.average(images, axis=0)

    # Initialise the parameters
    etas = [0, 1]
    s = np.random.randint(2, size=avg_image.shape)
    pi = np.random.rand(4)

    iterations = 0
    epsilon = 100
    while iterations < 150 and epsilon > 1e-5:
        alpha = posterior_pose_probs(images, etas, s, pi)
        s, new_etas, pi = assignment2_m_step(images, etas, alpha)
        epsilon = np.sum(new_etas) - np.sum(etas)
        etas = new_etas
        iterations += 1

    print("Etas :", etas)
    print("Pi :", pi)
    plt.imshow(s, cmap='gray')
    plt.title("Estimated shape")
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    Assignment1()
    Assignment2()
