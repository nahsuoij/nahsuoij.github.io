#!/usr/bin/env python3
"""
Generate character-colored frames.js from badapple.mp4.
Character colors are applied ONLY to silhouette edge/border pixels.
The silhouette core remains white (ID 1).
"""
import cv2
import numpy as np
import base64
import zlib
import struct
import subprocess
import os

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


def load_original_edge_mapping():
    """
    Load d615a1e frames. For each frame, extract which character color IDs
    appear on the silhouette edges.
    """
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
        frames.append(grid.reshape(H, W))
        offset += rle_len * 2
    
    return frames


def build_edge_schedule(original_frames):
    """
    From original 900 frames, determine which character color appears
    on the edges in each frame. Returns per-frame dominant edge color.
    """
    frame_edge_colors = []
    for grid in original_frames:
        # Find edge pixels: non-black, non-white
        edge_colors = [c for c in set(grid.flatten()) if c >= 2]
        if not edge_colors:
            frame_edge_colors.append(1)  # default white
        elif len(edge_colors) == 1:
            frame_edge_colors.append(edge_colors[0])
        else:
            # Pick the one with most edge pixels
            counts = {c: (grid == c).sum() for c in edge_colors}
            frame_edge_colors.append(max(counts, key=counts.get))
    
    return frame_edge_colors


def get_edge_color(frame_idx, frame_edge_colors):
    """Get character edge color for a frame, with cyclic extension."""
    orig_len = len(frame_edge_colors)
    if frame_idx < orig_len:
        return frame_edge_colors[frame_idx]
    # Cyclic: map back into original range
    lookup = frame_idx % orig_len
    return frame_edge_colors[lookup]


def process_frame(gray, edge_color_id):
    """
    Convert grayscale frame to character-colored grid.
    - Black pixels → ID 0
    - White core pixels → ID 1
    - Edge/border pixels → edge_color_id
    """
    # Binarize
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Find edge pixels using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    edge_mask = binary - eroded  # pixels that are on the border
    
    # Build grid
    grid = np.zeros(H * W, dtype=np.uint8)
    binary_flat = binary.flatten()
    edge_flat = edge_mask.flatten()
    
    # Background = 0 (already initialized)
    # Core = white (1)
    grid[binary_flat > 0] = 1
    # Edge = character color
    grid[edge_flat > 0] = edge_color_id
    
    return grid


def rle_encode(grid):
    flat = grid.flatten()
    rle = []
    i = 0
    while i < len(flat):
        val = flat[i]
        count = 1
        while i + count < len(flat) and flat[i + count] == val and count < 255:
            count += 1
        rle.append((count, int(val)))
        i += count
    return rle


def main():
    print("=== Bad Apple!! Edge-Colored Frame Generator ===\n")
    
    print("Step 1: Loading original 900-frame edge mapping...")
    original = load_original_edge_mapping()
    edge_schedule = build_edge_schedule(original)
    print(f"  {len(original)} frames loaded")
    
    # Show edge color distribution
    from collections import Counter
    dist = Counter(edge_schedule)
    print(f"  Edge color distribution:")
    for cid, count in dist.most_common():
        print(f"    ID {cid}: {count} frames ({count/len(edge_schedule)*100:.1f}%)")
    
    print("\nStep 2: Extracting video frames...")
    cap = cv2.VideoCapture(os.path.join(WORK_DIR, 'badapple.mp4'))
    video_frames = []
    step = VIDEO_FPS / TARGET_FPS
    idx = 0.0
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while int(idx) < total_vid:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        video_frames.append(gray)
        idx += step
    cap.release()
    print(f"  {len(video_frames)} frames extracted")
    
    print("\nStep 3: Generating edge-colored frames...")
    all_rles = []
    color_usage = Counter()
    
    for i, gray in enumerate(video_frames[:2629]):
        edge_color = get_edge_color(i, edge_schedule)
        grid = process_frame(gray, edge_color)
        rle = rle_encode(grid)
        all_rles.append(rle)
        
        for v in grid.flatten():
            color_usage[v] += 1
        
        if i % 500 == 0:
            edge_px = (grid >= 2).sum()
            white_px = (grid == 1).sum()
            print(f"  frame {i}: edge_color={edge_color}, edge_px={edge_px}, white_px={white_px}")
    
    total_px = sum(color_usage.values())
    print(f"\n  Pixel distribution:")
    for cid in sorted(color_usage.keys()):
        pct = color_usage[cid] / total_px * 100
        print(f"    ID {cid}: {color_usage[cid]} px ({pct:.1f}%)")
    
    print("\nStep 4: Packing...")
    data = bytearray()
    for rle in all_rles:
        data.extend(struct.pack('>H', len(rle)))
        for count, val in rle:
            data.extend(struct.pack('BB', count, val))
    
    compressed = zlib.compress(bytes(data), 9)
    b64data = base64.b64encode(compressed).decode('ascii')
    print(f"  Raw: {len(data)} bytes → Compressed: {len(compressed)} bytes → Base64: {len(b64data)} chars")
    
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
