import argparse
import logging
import time
from graph import build_graph, segment_graph, segment_graph_flow
from random import random
from PIL import Image, ImageFilter
import numpy as np
import cv2

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
    cv2.namedWindow('OpticalFlow', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)

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

        cv2.imshow('Segmented', segmented_frame)

        # Calculate optical flow
        flow_image = calculate_optical_flow(prev_frame, frame)
        cv2.imshow('OpticalFlow', flow_image)

        # Segment optical flow image
        flow_smooth = Image.fromarray(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB)).filter(ImageFilter.GaussianBlur(sigma))
        flow_smooth = np.array(flow_smooth).astype(int)

        logger.info("Creating graph for optical flow image...")
        flow_graph_edges = build_graph(flow_smooth, size[1], size[0], diff, neighbor == 8)

        logger.info("Merging graph for optical flow image...")
        #flow_forest = segment_graph(flow_graph_edges, size[0] * size[1], K, min_comp_size, threshold)
        flow_forest = segment_graph_flow(flow_smooth, graph_edges, size[0] * size[1], K, min_comp_size, threshold, diff,  size[1])

        logger.info("Visualizing segmentation for optical flow image...")
        flow_segmented = generate_image(flow_forest, size[1], size[0], rand = False)
        # Convert segmented image back to OpenCV format
        flow_segmented_frame = cv2.cvtColor(np.array(flow_segmented), cv2.COLOR_RGB2BGR)
        cv2.imshow('Segmented flow', flow_segmented_frame)

        # Combine original, original segmentation, optical flow, and optical flow segmentation
        combined_frame = np.zeros((2 * size[1], 2 * size[0], 3), dtype=np.uint8)
        combined_frame[:size[1], :size[0]] = frame
        combined_frame[:size[1], size[0]:] = segmented_frame
        combined_frame[size[1]:, :size[0]] = flow_image
        combined_frame[size[1]:, size[0]:] = flow_segmented_frame

        cv2.imshow('Combined', combined_frame)
        out.write(combined_frame)

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
