import numpy as np
import scipy

def normalize(v):
    """
    Normalizes 1D (possibly complex) vector
    :param v: vector to normalize
    :return: normalized vector
    """
    return v / np.sqrt(v.dot(v.conjugate()))

def gram_schmidt(B):
    """
    Gram Schmidt algorithm used to orthonormalize matrix B.
    :param B: complex matrix. e.g. \n
              B = [[b_{0,0}],[b_{0,1}],
                  [[b_{1,0}],[b_{1,1}]]\n where b_{i,j} corresponds
              to the i-th element of column basis vector j
              and the goal is to orthonormalize the basis spanned by
              the columns of B
    :return: Orthonormalized matrix with same dimensions as B
    """
    B[:, 0] = normalize(B[:, 0])
    for i in range(1, B.shape[1]):
        Ai = B[:, i]
        for j in range(0, i):
            Aj = B[:, j]
            t = Ai.dot(Aj.conjugate())
            Ai = Ai - t * Aj
        B[:, i] = normalize(Ai)
    return B


def factors(n):
    """
    Returns all factors of n
    """
    import functools
    return list(functools.reduce(list.__add__,
                                 ([i, n // i] for i in range(1, int(n ** 0.5) + 1)
                                  if n % i == 0)))


def find_intersect_line_ellipse(slope, radius_x, radius_y, theta=0):
    """
    Finds the intersection points (x1, y1) and (x2, y2) of a line
    going through the origin and an ellipse with radius_x, radius_y and angle with x axis theta
    """
    x1 = 1 / np.sqrt(
        (np.cos(theta) - slope * np.sin(theta)) ** 2 / radius_x ** 2 + (
                    np.sin(theta) + slope * np.cos(theta)) ** 2 / radius_y ** 2)
    x2 = -x1
    y1 = slope * x1
    y2 = slope * x2
    return np.array([x1, y1]), np.array([x2, y2])


def get_ellipse_radii_and_rotation(cov):
    """
    Computes radii (from sqrt eigenvalues) and rotation angle:
    returns radius_x, radius_y, -rotation_angle
    """

    if np.ndim(cov) == 1 or len(cov.flatten()) == 1:
        cov = np.diag([cov.flatten()[0]] * 2)

    (eigs, eigv) = scipy.linalg.eigh(cov)
    theta = np.arctan2(*eigv[:, 0])
    return np.sqrt(eigs[0]), np.sqrt(eigs[1]), -theta


def slope(p1, p2=(0,0)):
    return (p2[1] - p1[1] ) / (p2[0] - p1[0])


def vp_to_dbm(vp, z0=50):
    """Convert from signal peak voltage to power in dBm"""
    vrms = vp/np.sqrt(2)
    pwatt = vrms**2 / z0
    return 10 * np.log10(pwatt/1e-3)


def dbm_to_vp(dbm, z0=50):
    """Convert from signal power in dBm to peak voltage"""
    pwatt = 1e-3 * 10**(dbm/10)
    vrms = np.sqrt(pwatt * z0)
    return vrms * np.sqrt(2)


def kron(*args):
        """
        Compute the Kronecker product of multiple matrices or vectors.

        Args:
            *args: Variable number of input matrices.

        Returns:
            numpy.ndarray: The Kronecker product of the input matrices.

        Example:
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])
            C = np.array([[9, 10], [11, 12]])

            result = kronecker(A, B, C)

        The Kronecker product of multiple matrices is obtained by taking the tensor product
        of their vector spaces. If A, B, C, ..., are matrices with dimensions (m x n), (p x q),
        (r x s), ..., respectively, the resulting matrix will have dimensions (m * p * r * ... x
        n * q * s * ...). The basis elements of the resulting matrix can be represented using
        braket notation as:

        |aᵢ⟩ ⊗ |bⱼ⟩ ⊗ |cₖ⟩ ⊗ ...

        where |aᵢ⟩ represents the i-th basis element of matrix A, |bⱼ⟩ represents the j-th
        basis element of matrix B, |cₖ⟩ represents the k-th basis element of matrix C, and so on.

        The resulting matrix is obtained by taking the Kronecker product of each combination
        of basis elements from the input matrices in the given order.
        """
        result = args[0]
        for i in range(1, len(args)):
            result = np.kron(result, args[i])
        return result