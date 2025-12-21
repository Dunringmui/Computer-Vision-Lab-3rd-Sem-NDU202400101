import argparse
import os
import sys

import cv2
import numpy as np


def run_harris(img, blockSize=2, ksize=3, k=0.04, thresh_ratio=0.01):
    """Run Harris corner detector and return an image with corners marked and number of corners."""
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    dst = cv2.cornerHarris(gray_f, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)

    thresh = thresh_ratio * dst.max()
    yx = np.argwhere(dst > thresh)

    harris_img = original.copy()
    # Mark corners with small red circles
    for (y, x) in yx:
        cv2.circle(harris_img, (int(x), int(y)), 3, (0, 0, 255), -1)

    return harris_img, len(yx)


def run_sift(img, nfeatures=0):
    """Run SIFT detector (requires opencv-contrib) and return drawn keypoints and count."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = None
    try:
        sift = cv2.SIFT_create(nfeatures=nfeatures)
    except AttributeError:
        # Try legacy xfeatures2d
        if hasattr(cv2, 'xfeatures2d'):
            sift = cv2.xfeatures2d.SIFT_create(nfeatures)

    if sift is None:
        raise RuntimeError("SIFT is not available. Install opencv-contrib-python and try again.")

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    sift_img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_img, len(keypoints)


def join_hstack(images, bg_color=(255, 255, 255)):
    """Horizontally stack images of possibly different heights by padding."""
    heights = [im.shape[0] for im in images]
    max_h = max(heights)
    padded = []
    for im in images:
        h, w = im.shape[:2]
        if h < max_h:
            delta = max_h - h
            pad_top = delta // 2
            pad_bottom = delta - pad_top
            im = cv2.copyMakeBorder(im, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=bg_color)
        padded.append(im)
    return np.hstack(padded)


def main():
    parser = argparse.ArgumentParser(description='Harris and SIFT corner detection demo')
    parser.add_argument('image', nargs='?', default='apple.jpg', help='Path to input image (save the attached image as apple.jpg)')
    parser.add_argument('--method', choices=['harris', 'sift', 'both'], default='both', help='Detection method to run')
    parser.add_argument('--harris-thresh', type=float, default=0.01, help='Harris threshold (ratio of max corner response)')
    parser.add_argument('--harris-k', type=float, default=0.04, help='Harris free parameter k')
    parser.add_argument('--harris-blocksize', type=int, default=2, help='Harris blockSize parameter')
    parser.add_argument('--harris-ksize', type=int, default=3, help='Harris ksize (aperture)')
    parser.add_argument('--sift-nfeatures', type=int, default=0, help='SIFT nfeatures parameter')
    parser.add_argument('--no-show', action='store_true', help="Don't show GUI windows (headless/server use)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: image '{args.image}' not found in current directory: {os.getcwd()}")
        print("Please save the provided image as 'apple.jpg' in this folder or pass the path to your image file.")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image '{args.image}'.")
        sys.exit(1)

    outputs = []

    if args.method in ('harris', 'both'):
        harris_img, count_h = run_harris(img, blockSize=args.harris_blocksize, ksize=args.harris_ksize, k=args.harris_k, thresh_ratio=args.harris_thresh)
        harris_path = 'output_harris.jpg'
        cv2.imwrite(harris_path, harris_img)
        print(f"Harris: detected {count_h} corner candidates -> saved '{harris_path}'")
        outputs.append(harris_img)

    if args.method in ('sift', 'both'):
        try:
            sift_img, count_s = run_sift(img, nfeatures=args.sift_nfeatures)
            sift_path = 'output_sift.jpg'
            cv2.imwrite(sift_path, sift_img)
            print(f"SIFT: detected {count_s} keypoints -> saved '{sift_path}'")
            outputs.append(sift_img)
        except RuntimeError as e:
            print(f"SIFT error: {e}")
            print("To enable SIFT, install opencv-contrib-python: pip install opencv-contrib-python")

    # Create combined comparison image: Original | Harris | SIFT (when available)
    to_stack = [img]
    # ensure we only include unique items
    for o in outputs:
        to_stack.append(o)

    combined = join_hstack(to_stack)
    combined_path = 'corner_detection_results.jpg'
    cv2.imwrite(combined_path, combined)
    print(f"Combined results saved to '{combined_path}'")

    if not args.no_show:
        # Show the image in a resizable window
        cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
        cv2.imshow('Results', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()