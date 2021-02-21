from scipy.spatial import distance


def get_rect_vertices(rectangle):

    # Returns the vertices of a rectangle

    x1 = rectangle.left()
    y1 = rectangle.top()
    x2 = rectangle.right()
    y2 = rectangle.bottom()

    return [x1, x2, y1, y2]


def get_point_coords(point):

    # Returns the Co-ordinates of a given point

    x = point.x
    y = point.y

    return [x, y]


def calculate_ear(eye):

    # Calculates the EAR for an eye
    """
    http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

    Paper by Soukupová and Čech

    This paper states that eye blinks can be detected by calculating the Eye Aspect Ratio (EAR).
    It says that EAR can be computed by:

    EAR = (||p2 - p6|| + ||p3 - p5||)/(2||p1-p4||)
    """

    # d1 = || p2 - p6 ||
    d1 = distance.euclidean(eye[1], eye[5])

    # d2 = || p3 - p5 ||
    d2 = distance.euclidean(eye[2], eye[4])

    # d3 = || p1 - p4 ||
    d3 = distance.euclidean(eye[0], eye[3])

    ear = (d1 + d2) / (2 * d3)

    return ear
