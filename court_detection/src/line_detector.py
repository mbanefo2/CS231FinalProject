import cv2
import numpy as np
import itertools

drawn_lines = []
SHUTTLECOCK_POINT = (500, 430) # Random point to indicate shuttlecock position on image

def luminance(img: np.array) -> np.array:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 1]


def whiteline_detection(
    lum_img: np.array,
    val_th: int,
    diff_th: int,
    rng: int
) -> np.array:
    def tmp(arr, x: int, y: int, val_th, diff_th, rng) -> bool:
        if arr[x, y] > val_th:
            if arr[x, y] - arr[x-rng, y] > diff_th \
                    and arr[x, y] - arr[x+rng, y] > diff_th:
                return True
            elif arr[x, y] - arr[x, y-rng] > diff_th \
                    and arr[x, y] - arr[x, y+rng] > diff_th:
                return True
            else:
                return False
        else:
            return False

    ret = np.zeros(lum_img.shape, dtype=np.uint8)
    w = lum_img.shape[0]
    h = lum_img.shape[1]
    for i, j in itertools.product(range(rng, w-rng), range(rng, h-rng)):
        ret[i, j] = int(tmp(lum_img, i, j, val_th, diff_th, rng)) * 255
    return ret


def hough(white):
    return cv2.HoughLines(white, 1, np.pi / 180, 250)


def draw_lines(img, lines):

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1200*(-b))
        y1 = int(y0 + 1200*(a))
        x2 = int(x0 - 1200*(-b))
        y2 = int(y0 - 1200*(a))

        drawn_lines.append([(x1, y1), (x2, y2)])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


    intersection_points = find_intersection_points(drawn_lines)
    # print(intersection_points)

    # Annotate all intersections on court and print the coordinates on image
    for point in intersection_points:
        cv2.putText(img, f'({point[0]}, {point[1]})', point, fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 0.3,
        color = (0, 0, 0),
        thickness = 1)
        
    # Code to show random shuttlecock point on image. Feel free to comment out to see position.
    #   cv2.putText(img, f'({point[0]}, {point[1]})', SHUTTLECOCK_POINT, fontFace = cv2.FONT_HERSHEY_DUPLEX,
    #   fontScale = 0.3,
    #   color = (0, 255, 0),
    #   thickness = 3)
    
    # Code to automatically get the four outer coordinates of teh court
    # court_boundary_cordinates = farthest_points(intersection_points)

    # Mnaually handpicked edges of court based on the annotated values Feel free to comment out to see positions.
    # POINT_1 = (256, 572)
    # POINT_2 = (1001, 572)
    # POINT_3 = (381, 261)
    # POINT_4 = (879 ,261)
        
    # cv2.line(img, POINT_1, POINT_2, (0, 255, 0), 2)
    # cv2.line(img, POINT_1, POINT_3, (0, 255, 0), 2)
    # cv2.line(img, POINT_3, POINT_4, (0, 255, 0), 2)
    # cv2.line(img, POINT_2, POINT_4, (0, 255, 0), 2)

    # Check if shuttlecock is within the bunds of the court
    # inside = point_in_polygon((SHUTTLECOCK_POINT[0], SHUTTLECOCK_POINT[1]), [POINT_1, POINT_2, POINT_3, POINT_4])
    
    # answer = 'Yes' if inside == True else 'No'
    
    # Print our answer on the image. Yes indicates shuttlecock is within the court bounds and No indicates otherwise
    # cv2.putText(img, f'({answer})', (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX,
    # fontScale = 0.3,
    # color = (0, 255, 0),
    # thickness = 1)

    return img


def intersection_point(line1, line2):
    """Calculate the intersection point of two lines."""
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if den == 0:
        return None  # Lines are parallel

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

    if not (0 <= ua <= 1 and 0 <= ub <= 1):
        return None  # Intersection point is outside the line segments

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)

    return (int(x), int(y))


def find_intersection_points(lines):
    """Find the intersection points of all lines."""
    intersection_points = set()

    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            point = intersection_point(line1, line2)
            if point:
                intersection_points.add(point)

    return intersection_points



def farthest_points(coordinates):
    max_distance = 0
    farthest_points = None
    for combination in itertools.combinations(coordinates, 4):
        distance_matrix = [[((x1-x2)**2 + (y1-y2)**2)**0.5 for (x2, y2) in combination] for (x1, y1) in combination]
        max_distance_in_combination = max([max(row) for row in distance_matrix])
        if max_distance_in_combination > max_distance:
            max_distance = max_distance_in_combination
            farthest_points = combination
    return farthest_points


def point_in_polygon(point, polygon):
    x, y = point
    intersections = 0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if y1 == y2:  # horizontal edge, ignore
            continue
        if min(y1, y2) <= y <= max(y1, y2):
            # compute x coordinate of intersection
            x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x_intersect > x:
                intersections += 1

    return intersections % 2 == 1


if __name__ == "__main__":
    import sys
    finename = sys.argv[1]
    img = cv2.imread(finename)
    width, height =  1280, 720
    img = cv2.resize(img, (width, height))
    img = cv2.bilateralFilter(img, 15, 20, 20)
    img = cv2.bilateralFilter(img, 15, 20, 20)
    img = cv2.bilateralFilter(img, 15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    img = cv2.bilateralFilter(img,15, 20, 20)
    lum = luminance(img)

    print(img.shape)
    print(lum.shape)

    white = whiteline_detection(lum, 150, 25, 3)
    hough_lines = hough(white)
    hough_ = draw_lines(img, hough_lines)

    for i in [lum, white, hough_]:

        cv2.imshow('image', i)
        
        key = cv2.waitKey(0)

        if key == 27: 
            cv2.destroyAllWindows()
    
