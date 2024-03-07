import argparse
import logging
import time
from graph import build_graph, segment_graph, segment_graph_flow
from random import random
from PIL import Image, ImageFilter
import numpy as np
import cv2

#-------------------------------------------------------------------
from enum import Enum

good_features_parameters = dict(maxCorners=200, qualityLevel=0.1, minDistance=1, blockSize=3, useHarrisDetector=True, k=0.04)
optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)


class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)
    WHITE = (255,255,255)
    BLACK = (0, 0, 0)

def wait_until_space_pressed():
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Check if the space key is pressed
            break
            
def ransac_vanishing_point(edgelets, height, num_ransac_iter=2000, threshold_inlier=5):
    """Estimate vanishing point using Ransac.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.

    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.

    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)
    #print("lines :", lines)

    num_pts = strengths.size
    #print("num_pts :", num_pts)

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)
        #print(current_model)

        if np.sum(current_model**2) < 1 or current_model[2] == 0: # or current_model[1] / current_model[2] > height/2:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            # logging.info("Current best model has {} votes at iteration {}".format(
                # current_votes.sum(), ransac_iter))

    return best_model

def vis_edgelets(image, edgelets, color=cv_colors.RED.value, output_path="vis_image.png", show=True):
    """Helper function to visualize edgelets using OpenCV and optionally store the result to a file."""
    
    locations, directions, strengths = edgelets
    print("locations.shape: ", locations.shape)

    # Create a copy of the image to draw on
    vis_image = np.copy(image)

    for i in range(locations.shape[0]):
        # Calculate start and end points for each edgelet
        start_point = (int(locations[i, 0] - directions[i, 0] * strengths[i] / 2),
                       int(locations[i, 1] - directions[i, 1] * strengths[i] / 2))
        end_point = (int(locations[i, 0] + directions[i, 0] * strengths[i] / 2),
                     int(locations[i, 1] + directions[i, 1] * strengths[i] / 2))

        # Draw the edgelet on the image
        cv2.line(vis_image, start_point, end_point, color=color, thickness=1)

    # Display the image
    if show:
        cv2.imshow('Edgelets', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the result to a file if the output_path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Result saved to: {output_path}")
        

def compute_edgelets(prev_points, next_points, box1, box2, scale=5, threshold=3.0):

    flow_vectors = next_points - prev_points

    print("Shapes before filtering:")
    print("prev_points shape:", prev_points.shape)
    print("next_points shape:", next_points.shape)
    print("flow_vectors shape:", flow_vectors.shape)

    # Filter flow vectors below the threshold
    magnitude = np.linalg.norm(flow_vectors, axis=1)
    mask = magnitude >= threshold

    # Filter points within box1
    mask_box1 = (
        (prev_points[:, 0] >= box1[0]) & (prev_points[:, 0] <= box1[2]) &
        (prev_points[:, 1] >= box1[1]) & (prev_points[:, 1] <= box1[3])
    )

    # Filter points within box2
    mask_box2 = (
        (next_points[:, 0] >= box2[0]) & (next_points[:, 0] <= box2[2]) &
        (next_points[:, 1] >= box2[1]) & (next_points[:, 1] <= box2[3])
    )

    mask = mask & mask_box1 & mask_box2

    prev_points = prev_points[mask]
    next_points = next_points[mask]
    flow_vectors = flow_vectors[mask]

    print("Shapes after filtering:")
    print("prev_points shape:", prev_points.shape)
    print("next_points shape:", next_points.shape)
    print("flow_vectors shape:", flow_vectors.shape)

    lines = np.concatenate([prev_points[:, None], next_points[:, None]], axis=1)
    lines = lines.reshape(-1, 2, 2)
    #lines = np.int32(lines + 0.5)

    # Uncomment the following lines to draw the lines on the frame
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    locations = []
    directions = []
    strengths = []

    for (x1, y1), (x2, y2) in lines:
        p0, p1 = np.array([x1, y1]), np.array([x2, y2])
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize if directions is not empty
    if directions:
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    return locations, directions, strengths



def vis_model(image, edgelets, model, output_path="vis_model.png", show=True):
    """Helper function to visualize computed model using OpenCV and optionally store the result to a file."""

    # Create a copy of the image to draw on
    vis_image = np.copy(image)

    # Visualize edgelets with green color
    vis_edgelets(vis_image, edgelets, color=(0, 255, 0))

    # Get inliers based on the computed votes
    inliers = compute_votes(edgelets, model, 10) > 0

    # Extract inlier edgelets
    edgelets = (edgelets[0][inliers], edgelets[1][inliers], edgelets[2][inliers])
    locations, directions, strengths = edgelets

    # Visualize inlier edgelets with red color
    vis_edgelets(vis_image, edgelets, color=(0, 0, 255))

    # Get vanishing point in homogeneous coordinates
    vp = model / model[2]

    # Draw vanishing point as a blue circle
    cv2.circle(vis_image, (int(vp[0]), int(vp[1])), 5, (255, 0, 0), -1)

    # Draw lines from edgelet locations to vanishing point
    for i in range(locations.shape[0]):
        cv2.line(vis_image, (int(locations[i, 0]), int(locations[i, 1])),
                 (int(vp[0]), int(vp[1])), (255, 0, 0), 2)

    # Display the image
    if show:
        cv2.imshow('Model Visualization', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the result to a file if the output_path is provided
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Result saved to: {output_path}")


def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.

    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines
    

def compute_votes(edgelets, model, threshold_inlier=5):
    """Compute votes for each of the edgelet against a given vanishing point.

    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.

    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.

    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    
    theta = np.zeros_like(cosine_theta)
    
    # Iterate element by element to compute theta
    for i in range(len(cosine_theta)):
        theta[i] = np.arccos(np.abs(cosine_theta[i])) if cosine_theta[i] < 1 and cosine_theta[i] > -1 else threshold_inlier * np.pi / 180
    
    #theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths



#___________________________________________________________________________




def diff(img, x1, y1, x2, y2):
    _out = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    return np.sqrt(_out)

def threshold(size, const):
    return (const * 1.0 / size)

def generate_image(forest, width, height, rand = True):
    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in range(width*height)]

    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            if rand:
                im[x, y] = colors[comp]
            else:
                im[x, y] =  tuple(forest.color(comp).astype(int))

    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)

def calculate_optical_flow(prev_frame, next_frame):
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_image

def get_segmented_video(sigma, neighbor, K, min_comp_size, input_file, output_file):
    if neighbor != 4 and neighbor != 8:
        logger.warn('Invalid neighborhood chosen. The acceptable values are 4 or 8.')
        logger.warn('Segmenting with 4-neighborhood...')
    start_time = time.time()

    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (2*size[0], 2*size[1]))

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Segmented', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('OpticalFlow', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)

    ret, prev_frame = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Original', frame)

        # Convert frame to PIL Image
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Gaussian Filter
        smooth = pil_frame.filter(ImageFilter.GaussianBlur(sigma))
        smooth = np.array(smooth).astype(int)

        logger.info("Creating graph...")
        graph_edges = build_graph(smooth, size[1], size[0], diff, neighbor == 8)

        logger.info("Merging graph...")
        forest = segment_graph(smooth, graph_edges, size[0] * size[1], K, min_comp_size, threshold,  size[1])

        logger.info("Visualizing segmentation and saving into: {}".format(output_file))
        image = generate_image(forest, size[1], size[0], rand = False)
        # Convert segmented image back to OpenCV format
        segmented_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


        
        # Segment optical flow image
        # flow_smooth = Image.fromarray(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB)).filter(ImageFilter.GaussianBlur(sigma))
        # flow_smooth = np.array(flow_smooth).astype(int)

        # Calculate optical flow
        # flow_image = calculate_optical_flow(prev_frame, frame)
        # cv2.imshow('OpticalFlow', flow_image)
        
        pimg, cimg = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_points = cv2.goodFeaturesToTrack(pimg, mask=None, **good_features_parameters)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(pimg, cimg, prev_points, None, **optical_flow_parameters)
        # Selects good feature points for previous position
        good_old = prev_points[status == 1].astype(int)
        # Selects good feature points for next position
        good_new = next_points[status == 1].astype(int)
        
        # Draw tracked points on images
        for i, (prev_point, next_point) in enumerate(zip(good_old.reshape(-1, 2), good_new.reshape(-1, 2))):
            color = (0, 0, 255)  # Red color
            cv2.circle(segmented_frame, tuple(map(int, next_point)), 3, color, -1)  # Draw circle on image 2

 
        
        components = forest.get_components()
        #print("components :", components)
        
        for cmp in components:
            box = forest.nodes[cmp].box
            
            box_swapped = [box[1], box[0], box[3], box[2]]
            #cv2.rectangle(segmented_frame, box_swapped[:2], box_swapped[2:], (0, 255, 255), 1)
            

            edgelets1 = compute_edgelets(good_old.reshape(-1, 2), good_new.reshape(-1, 2), box_swapped, box_swapped)
            if edgelets1[1].shape[0] > 0:
                # Visualize the edgelets
                vis_edgelets(segmented_frame, edgelets1)
                vp1 = ransac_vanishing_point(edgelets1, segmented_frame.shape[0],  2000, threshold_inlier=5)
                print("vp :", vp1)
                vis_model(segmented_frame, edgelets1, vp1, output_path = f"vis_model_{ind+1}.png")
                vp = vp1 / vp1[2]
                print(f"vp ({ind+1}) : {vp}")
            
        cv2.imshow('Segmented', segmented_frame)

        # logger.info("Creating graph for optical flow image...")
        # flow_graph_edges = build_graph(flow_smooth, size[1], size[0], diff, neighbor == 8)

        # logger.info("Merging graph for optical flow image...")
        # #flow_forest = segment_graph(flow_graph_edges, size[0] * size[1], K, min_comp_size, threshold)
        # flow_forest = segment_graph_flow(flow_smooth, graph_edges, size[0] * size[1], K, min_comp_size, threshold, diff,  size[1])

        # logger.info("Visualizing segmentation for optical flow image...")
        # flow_segmented = generate_image(flow_forest, size[1], size[0], rand = False)
        # # Convert segmented image back to OpenCV format
        # flow_segmented_frame = cv2.cvtColor(np.array(flow_segmented), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Segmented flow', flow_segmented_frame)

        # Combine original, original segmentation, optical flow, and optical flow segmentation
        # combined_frame = np.zeros((2 * size[1], 2 * size[0], 3), dtype=np.uint8)
        # combined_frame[:size[1], :size[0]] = frame
        # combined_frame[:size[1], size[0]:] = segmented_frame
        # combined_frame[size[1]:, :size[0]] = flow_image
        # combined_frame[size[1]:, size[0]:] = flow_segmented_frame

        # cv2.imshow('Combined', combined_frame)
        # out.write(combined_frame)

        prev_frame = frame

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    logger.info('Number of components: {}'.format(forest.num_sets))
    logger.info('Total running time: {:0.4}s'.format(time.time() - start_time))

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='Graph-based Video Segmentation')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='a float for the Gaussian Filter')
    parser.add_argument('--neighbor', type=int, default=8, choices=[4, 8],
                        help='choose the neighborhood format, 4 or 8')
    parser.add_argument('--K', type=float, default=10.0,
                        help='a constant to control the threshold function of the predicate')
    parser.add_argument('--min-comp-size', type=int, default=100,
                        help='a constant to remove all the components with fewer number of pixels')
    parser.add_argument('--input-file', type=str, default="./assets/seg_test.mp4",
                        help='the file path of the input video')
    parser.add_argument('--output-file', type=str, default="./assets/seg_test_out_combined.mp4",
                        help='the file path of the output video')
    args = parser.parse_args()

    # basic logging settings
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logger = logging.getLogger(__name__)

    get_segmented_video(args.sigma, args.neighbor, args.K, args.min_comp_size, args.input_file, args.output_file)
