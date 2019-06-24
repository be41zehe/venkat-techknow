import numpy as np
import time
import icp
from plyfile import PlyData, PlyElement

# Constants
N = 10                                    # number of random points in the dataset
num_tests = 50                             # number of test iterations
dim = 1                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set
tolerance = 0.001


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
##    b, c, d = -axis*np.sin(theta/2.)
##
##    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
##                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
##                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    return np.array(a)


def test_best_fit():

    # Generate a random dataset
    #A = np.random.rand(N, dim)
    A=np.array([])
    # Generate a random dataset
    plydata = PlyData.read('bun045.ply')
    # Generate a random dataset
    for j in range(0, plydata.elements[0].count):
        A=np.append(A,plydata['vertex'][j][0])
    A=np.reshape(A,(plydata.elements[0].count,1))

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        #B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((A.size, 2))
        C[:,0:1] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:1], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    #print('best fit time: {:.3}'.format(total_time/num_tests))
    print('T_Bestfit\n',T)
    print('R_Bestfit\n',R)

    return


def test_icp():

    # Generate a random dataset
    #A = np.random.rand(N, dim)
    A=np.array([])
    # Generate a random dataset
    plydata = PlyData.read('bun045.ply')
    # Generate a random dataset
    for j in range(0, plydata.elements[0].count):
        A=np.append(A,plydata['vertex'][j][0])
    A=np.reshape(A,(plydata.elements[0].count,1))

    total_time = 0


    B = np.copy(A)

    # Translate
    t = np.random.rand(dim)*translation
    B += t

    # Rotate
    R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
    B = np.dot(R, B.T).T

    # Add noise
    #B += np.random.randn(N, dim) * noise_sigma

    # Shuffle to disrupt correspondence
    np.random.shuffle(B)

    # Run ICP
    start = time.time()
    T, distances, iterations = icp.icp(B, A, None, num_tests, tolerance)
    total_time += time.time() - start

    # Make C a homogeneous representation of B
    C = np.ones((A.size, 2))
    C[:,0:1] = np.copy(B)

    # Transform C
    C = np.dot(T, C.T).T

    assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
    assert np.allclose(T[0:1,0:1].T, R, atol=6*noise_sigma)     # T and R should be inverses
    assert np.allclose(-T[0:1,1], t, atol=6*noise_sigma)        # T and t should be inverses

    #print('icp time: {:.3}'.format(total_time/num_tests))
    print('T_ICP\n',T)
    print('R_ICP\n',R)


    return


if __name__ == "__main__":
    test_best_fit()
    test_icp()
