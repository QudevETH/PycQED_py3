import logging
log = logging.getLogger(__name__)

import numpy as np
import itertools as it


# geometry functions
def getLine(p1, p2, eps=1e-30):
    """
    p1 is a tuple of the first point
    p2 is a tuple of the second point
    returns a tuple of the slope and y-intercept of the line going throug
    both points
    """
    if np.abs(p1[0] - p2[0]) < eps:
        slope = 1/eps
    else:
        slope = float((p1[1] - p2[1]) / (p1[0] - p2[0]))
    yint = float((-1 * (p1[0])) * slope + p1[1])
    return slope, yint


def getIntersection(line1, line2):
    """
    line1 is a tuple of m and b of the line in the form y=mx+b
    line2 is a tuple of m and b of the line in the form y=mx+b
    returns a tuple of the points of the intersection of the two lines
    """
    slope1, slope2 = line1[0], line2[0]
    yint1, yint2 = line1[1], line2[1]
    matA = np.matrix(str(-1 * slope1) + ' 1;' + str(-1 * slope2) + ' 1')
    matB = np.matrix(str(yint1) + '; ' + str(yint2))
    invA = matA.getI()
    resultant = invA * matB
    return resultant[0,0], resultant[1,0]


def perpSlope(slope, eps=1e-30):
    # takes slope(float) and returns the slope of a line perpendicular to it
    if np.abs(slope) < eps:
        return 1/eps
    return (slope * -1) ** -1


def lineFromSlope(slope, point):
    """
    slope is a float of slope
    point is a tuple of ...
    returns tuple of slope and y intercept
    """
    return slope, ((slope * (-1 * point[0])) + point[1])


def find_heights_and_intersections(points, eps=1e-10):
    """
    Find heights of triangle and intersection with opposite side
    """
    sides = []
    heights = []
    height_vertices = []
    intersections = []
    if isinstance(points, np.ndarray):
        points = points.tolist()
    for p in it.combinations(points, 2):
        side = getLine(p[0], p[1])
        sides.append(side)
        perp_slope = perpSlope(side[0])
        h_v = [pt for pt in points if pt not in p][0]

        height = lineFromSlope(slope=perp_slope, point=h_v)
        heights.append(height)
        height_vertices.append(h_v)
    for h in heights:
        temp = []
        for s in sides:
            try:
                if np.abs(h[0]*s[0] + 1.) < eps:
                    temp.append(getIntersection(h, s))
                elif (np.abs(h[0]) < eps and np.abs(s[0]) > 1./eps) or \
                        (np.abs(s[0]) < eps and np.abs(h[0]) > 1./eps):
                    temp.append(getIntersection(h, s))
            except np.linalg.LinAlgError:
                print("No intersection / coincidence between height {} "
                      "and side {}".format(h,s))

        inter = temp
        assert len(inter) == 1
        intersections.append(inter[0])
    return sides, heights, height_vertices, intersections


def predict_pt_proba(pt, height_vertices, intersections, vertices):
    proba = []
    if isinstance(vertices, np.ndarray): # for ease of comparison
        vertices = vertices.tolist()
    for inter, h_v in zip(intersections, height_vertices):
        side_vert = [v for v in vertices if v != h_v][0] - np.array(h_v)
        inter_translated = np.array(inter) - np.array(h_v)
        pt_translated = np.array(pt) - np.array(h_v)
        inter_normalized = inter_translated/np.linalg.norm(
            inter_translated, ord=2, keepdims=True)
        proj = np.dot(pt_translated, inter_normalized) / \
               np.dot(inter_normalized, side_vert)
        proba.append(1 - proj)
    return proba


def predict_proba_avg_ro(X, cal_points):
    # Possible improvements: make for any number of cal points and channels,
    # remove (convenient) dependency on shapely
    """
    Predicts probabilities of each state of average readout data
    X using calibration points for 3 (fixed number) of calibration points
    Args:
        X: (n_samples, n_channels) average readout data points.
        cal_points: list or array of calibration points. Expects size of (3, 2):
            namely 3 calpoints (one for each state), and 2 channels
        proj_pt_on_triangle: whether or not to project points outside the state-
            defined triangle onto the triangle. False should be used
            for debugging only.
    returns
    """
    probas = []
    assert X.shape[1] == 2 and np.ndim(X) == 2, \
        f"For now measurement data should be of shape (any, 2)." + \
        f" Received shape: {X.shape}. More flexible method might be" + \
        f" implemented in the future"
    assert np.array(cal_points).shape == (3, 2), \
        f"For now calpoints should be of shape (3, 2)." + \
        f" Received shape: {np.array(cal_points).shape}." \
        f" More flexible method might be implemented in the future."
    sides, heights, height_vertices, intersections = \
        find_heights_and_intersections(cal_points)
    for pt in X:
        probas.append(predict_pt_proba(pt, height_vertices,
                                       intersections, cal_points))
    # reorder probability colulmns to give them back in same order as cal_points
    # could probably be done in a shorter way
    reordered_idx = []
    for i, c in enumerate(cal_points):
        for j, h in enumerate(height_vertices):
            if list(h) == list(c):
                reordered_idx.append(j)

    return np.array(probas)[:, reordered_idx]