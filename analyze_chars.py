#!/usr/bin/env python3
"""
Analyze Bad Apple video to detect character appearances by silhouette shape.
Outputs a per-frame character mapping for all 2629 frames.
"""
import cv2
import numpy as np
import os
import json
from collections import defaultdict

WORK_DIR = '/root/.openclaw/workspace/github-pages'
W, H = 48, 36
VIDEO_FPS = 15.0
TARGET_FPS = 12


def extract_all_frames():
    """Extract all frames from video at target FPS."""
    cap = cv2.VideoCapture(os.path.join(WORK_DIR, 'badapple.mp4'))
    frames = []
    step = VIDEO_FPS / TARGET_FPS
    idx = 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while int(idx) < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        frames.append(binary)
        idx += step
    cap.release()
    return frames


def compute_shape_features(binary_frame):
    """Extract shape features from a binary silhouette frame."""
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 50:
        return None
    
    # Bounding box
    x, y, bw, bh = cv2.boundingRect(contour)
    aspect = bw / max(bh, 1)
    
    # Hu moments (shape descriptor)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Solidity (area / convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    
    # Extent (area / bounding box area)
    extent = area / max(bw * bh, 1)
    
    # Number of convexity defects (fingers, wings, etc.)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
        num_defects = len(defects) if defects is not None else 0
    except:
        num_defects = 0
    
    # Vertical center of mass
    m00 = moments['m00']
    if m00 > 0:
        cy = moments['m10'] / m00 / max(bw, 1)  # normalized
        cx = moments['m01'] / m00 / max(bh, 1)
    else:
        cy, cx = 0.5, 0.5
    
    # Top-heavy vs bottom-heavy (height distribution)
    top_half = binary_frame[:H//2, :].sum()
    bottom_half = binary_frame[H//2:, :].sum()
    height_ratio = top_half / max(bottom_half, 1)
    
    # Left-right symmetry
    left_half = binary_frame[:, :W//2].sum()
    right_half = binary_frame[:, W//2:].sum()
    lr_symmetry = min(left_half, right_half) / max(left_half, right_half, 1)
    
    return {
        'area': area,
        'aspect': aspect,
        'hu': hu_log.tolist(),
        'solidity': solidity,
        'extent': extent,
        'num_defects': num_defects,
        'cx': cx, 'cy': cy,
        'height_ratio': height_ratio,
        'lr_symmetry': lr_symmetry,
        'bw': bw, 'bh': bh,
    }


def cluster_by_similarity(frames, sample_interval=10):
    """
    Cluster frames by silhouette similarity.
    Returns cluster assignments for each frame.
    """
    print("Computing shape features for sampled frames...")
    features = []
    for i, f in enumerate(frames):
        if i % sample_interval == 0:
            feat = compute_shape_features(f)
            features.append((i, feat))
        else:
            features.append((i, None))
    
    # Compute feature vectors for frames with features
    feat_vectors = []
    feat_indices = []
    for i, feat in features:
        if feat is not None:
            vec = [
                feat['area'] / 1000,
                feat['aspect'],
                feat['solidity'],
                feat['extent'],
                feat['num_defects'],
                feat['height_ratio'],
                feat['lr_symmetry'],
                feat['cx'],
                feat['cy'],
            ] + feat['hu'][:4]  # Use first 4 Hu moments
            feat_vectors.append(vec)
            feat_indices.append(i)
    
    feat_vectors = np.array(feat_vectors)
    
    # Normalize features
    mean = feat_vectors.mean(axis=0)
    std = feat_vectors.std(axis=0) + 1e-10
    feat_vectors_norm = (feat_vectors - mean) / std
    
    # Simple k-means-like clustering
    from scipy.spatial.distance import cdist
    
    # Use hierarchical clustering
    from scipy.cluster.hierarchy import fcluster, linkage
    
    print(f"Clustering {len(feat_vectors)} frame features...")
    Z = linkage(feat_vectors_norm, method='ward')
    clusters = fcluster(Z, t=15, criterion='maxclust')
    
    print(f"Found {len(set(clusters))} clusters")
    
    # For each cluster, show representative frames
    cluster_frames = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        frame_idx = feat_indices[idx]
        cluster_frames[cluster_id].append(frame_idx)
    
    return clusters, feat_indices, cluster_frames


def main():
    print("=== Bad Apple Character Analysis ===\n")
    
    print("Extracting video frames...")
    frames = extract_all_frames()
    print(f"  {len(frames)} frames\n")
    
    clusters, feat_indices, cluster_frames = cluster_by_similarity(frames)
    
    # For each cluster, show which time range it covers and sample frames
    print("\nCluster analysis:")
    for cid in sorted(cluster_frames.keys()):
        f_list = sorted(cluster_frames[cid])
        t_start = f_list[0] / TARGET_FPS
        t_end = f_list[-1] / TARGET_FPS
        count = len(f_list)
        
        # Sample a frame to show shape
        sample_idx = f_list[len(f_list)//2]
        sample = frames[sample_idx]
        contours, _ = cv2.findContours(sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
        else:
            bw, bh, area = 0, 0, 0
        
        print(f"  Cluster {cid}: {count} frames, {t_start:.1f}s-{t_end:.1f}s, "
              f"bbox={bw}x{bh}, area={area:.0f}")
        
        # Show ASCII art of representative frame
        if count > 0 and area > 100:
            print(f"    Sample frame {sample_idx}:")
            for r in range(0, H, 2):
                line = ''
                for c in range(0, W, 2):
                    if sample[r, c] > 0:
                        line += '#'
                    else:
                        line += '.'
                print(f"    {line}")


if __name__ == '__main__':
    main()
