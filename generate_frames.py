#!/usr/bin/env python3
"""
Generate character-colored frames.js from badapple.mp4.
Uses template matching against original 900-frame character data
to precisely detect which character appears in each frame.
"""
import cv2
import numpy as np
import base64
import zlib
import struct
import subprocess
import os
from collections import Counter

CHAR_COLORS = {
    0: [8, 8, 14],
    1: [255, 255, 255],
    2: [255, 120, 170],  # Reimu pink
    3: [255, 220, 50],   # Marisa yellow
    4: [200, 100, 255],  # Patchouli purple
    5: [255, 60, 60],    # Remilia red
    6: [140, 200, 255],  # Cirno blue
    7: [255, 180, 100],  # orange
    8: [180, 255, 180],  # green
    9: [255, 150, 200],  # light pink
    10: [220, 180, 255], # lavender
    11: [255, 255, 200], # cream accent
}

TARGET_FPS = 12
W, H = 48, 36
VIDEO_FPS = 15.0
WORK_DIR = '/root/.openclaw/workspace/github-pages'


def load_original_data():
    """Load d615a1e frames - returns list of (binary_mask, edge_color_id)."""
    src = subprocess.run(
        ['git', 'show', 'd615a1e:frames.js'],
        capture_output=True, text=True, cwd=WORK_DIR
    ).stdout
    
    start = src.index('FRAME_DATA = "') + len('FRAME_DATA = "')
    end = src.index('"', start)
    compressed = base64.b64decode(src[start:end])
    raw = zlib.decompress(compressed)
    
    frames = []
    offset = 0
    while offset < len(raw):
        rle_len = (raw[offset] << 8) | raw[offset + 1]
        offset += 2
        grid = np.zeros(H * W, dtype=np.uint8)
        idx = 0
        for i in range(rle_len):
            count = raw[offset + i * 2]
            val = raw[offset + i * 2 + 1]
            for _ in range(count):
                if idx < H * W:
                    grid[idx] = val
                    idx += 1
        
        grid_2d = grid.reshape(H, W)
        # Extract binary mask (any non-black pixel)
        binary = (grid_2d > 0).astype(np.uint8) * 255
        # Find dominant edge color
        edge_colors = [c for c in set(grid_2d.flatten()) if c >= 2]
        if edge_colors:
            # Pick the color with most pixels
            counts = {c: (grid_2d == c).sum() for c in edge_colors}
            edge_color = max(counts, key=counts.get)
        else:
            edge_color = 1
        
        frames.append((binary, edge_color))
        offset += rle_len * 2
    
    return frames


def extract_video_frames():
    """Extract frames from video at target FPS."""
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
        small = cv2.resize(frame, (W, H), cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        frames.append(binary)
        idx += step
    cap.release()
    return frames


def compute_signature(binary):
    """Compute a compact shape signature for fast comparison."""
    # Resize to 16x16 for fast comparison
    small = cv2.resize(binary, (16, 16), cv2.INTER_AREA)
    return small.flatten().astype(np.float32) / 255.0


def find_best_match(video_binary, original_signatures, original_data):
    """Find the best matching original frame for a video frame."""
    sig = compute_signature(video_binary)
    
    # Quick size check
    video_area = video_binary.sum() / 255
    
    best_score = float('inf')
    best_idx = 0
    
    for i, (orig_sig, (orig_binary, _)) in enumerate(zip(original_signatures, original_data)):
        orig_area = orig_binary.sum() / 255
        
        # Skip if area is too different
        area_ratio = min(video_area, orig_area) / max(video_area, orig_area, 1)
        if area_ratio < 0.3:
            continue
        
        # L2 distance on signatures
        dist = np.linalg.norm(sig - orig_sig)
        
        if dist < best_score:
            best_score = dist
            best_idx = i
    
    return best_idx, best_score


def main():
    print("=== Bad Apple!! Template-Matching Frame Generator ===\n")
    
    print("Step 1: Loading original 900-frame character data...")
    original_data = load_original_data()
    print(f"  {len(original_data)} frames loaded")
    
    # Compute signatures for original frames
    print("  Computing shape signatures...")
    original_signatures = [compute_signature(b) for b, _ in original_data]
    
    # Show original character distribution
    color_dist = Counter(c for _, c in original_data)
    print(f"  Original edge color distribution:")
    for cid, count in color_dist.most_common():
        print(f"    ID {cid}: {count} frames ({count/len(original_data)*100:.1f}%)")
    
    print("\nStep 2: Extracting video frames...")
    video_frames = extract_video_frames()
    print(f"  {len(video_frames)} frames extracted")
    
    print("\nStep 3: Template matching each frame...")
    all_rles = []
    match_stats = Counter()
    pixel_stats = Counter()
    
    for i, v_binary in enumerate(video_frames[:2629]):
        # Find best matching original frame
        match_idx, score = find_best_match(v_binary, original_signatures, original_data)
        _, edge_color = original_data[match_idx]
        
        # Build the grid: edge pixels get character color, core stays white
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(v_binary, kernel, iterations=1)
        edge_mask = v_binary - eroded
        
        grid = np.zeros(H * W, dtype=np.uint8)
        v_flat = v_binary.flatten()
        e_flat = edge_mask.flatten()
        
        grid[v_flat > 0] = 1        # white core
        grid[e_flat > 0] = edge_color  # character-colored edge
        
        # RLE encode
        rle = []
        j = 0
        flat = grid
        while j < len(flat):
            val = flat[j]
            count = 1
            while j + count < len(flat) and flat[j + count] == val and count < 255:
                count += 1
            rle.append((count, int(val)))
            j += count
        all_rles.append(rle)
        
        match_stats[edge_color] += 1
        for v in grid:
            pixel_stats[v] += 1
        
        if i % 500 == 0:
            print(f"  frame {i}: matched to original#{match_idx} (score={score:.3f}), edge_color={edge_color}")
    
    total_pixels = sum(pixel_stats.values())
    print(f"\n  Edge color distribution (all frames):")
    for cid in sorted(match_stats.keys()):
        print(f"    ID {cid}: {match_stats[cid]} frames ({match_stats[cid]/2629*100:.1f}%)")
    
    print(f"\n  Pixel distribution:")
    for cid in sorted(pixel_stats.keys()):
        print(f"    ID {cid}: {pixel_stats[cid]} px ({pixel_stats[cid]/total_pixels*100:.1f}%)")
    
    print("\nStep 4: Packing and compressing...")
    data = bytearray()
    for rle in all_rles:
        data.extend(struct.pack('>H', len(rle)))
        for count, val in rle:
            data.extend(struct.pack('BB', count, val))
    
    compressed = zlib.compress(bytes(data), 9)
    b64data = base64.b64encode(compressed).decode('ascii')
    print(f"  Raw: {len(data)} → Compressed: {len(compressed)} → Base64: {len(b64data)} chars")
    
    colors_json = '{' + ', '.join(f'"{k}": [{v[0]}, {v[1]}, {v[2]}]' for k, v in sorted(CHAR_COLORS.items())) + '}'
    
    output = f"""// Bad Apple!! — Character-mapped frame data
// {len(all_rles)} frames, {W}x{H} @ {TARGET_FPS}fps
// Character color palette
const CHAR_COLORS = {colors_json};
const FRAME_COUNT = {len(all_rles)};
const DOT_COLS = {W};
const DOT_ROWS = {H};
const FPS = {TARGET_FPS};
const FRAME_DATA = "{b64data}";"""
    
    with open(os.path.join(WORK_DIR, 'frames.js'), 'w') as f:
        f.write(output)
    
    print(f"\n✅ Generated frames.js — {len(all_rles)} frames, {W}x{H} @ {TARGET_FPS}fps")


if __name__ == '__main__':
    main()
