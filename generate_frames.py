#!/usr/bin/env python3
"""
Generate character-colored frames.js from badapple.mp4.
Uses original 900-frame character mapping as reference templates,
then extends to full 2629 frames via silhouette matching.
"""
import cv2
import numpy as np
import base64
import zlib
import struct
import subprocess
import json
import os

# Character color palette (from d615a1e)
CHAR_COLORS = {
    0: [8, 8, 14],      # black (background)
    1: [255, 255, 255],  # white (default silhouette)
    2: [255, 120, 170],  # pink (Reimu)
    3: [255, 220, 50],   # yellow (Marisa)
    4: [200, 100, 255],  # purple (Patchouli)
    5: [255, 60, 60],    # red (Remilia)
    6: [140, 200, 255],  # blue (Cirno)
    7: [255, 180, 100],  # orange
    8: [180, 255, 180],  # green
    9: [255, 150, 200],  # light pink
    10: [220, 180, 255], # lavender
    11: [255, 255, 200], # cream/white accent
}

TARGET_FPS = 12
TARGET_W = 48
TARGET_H = 36
VIDEO_FPS = 15.0
WORK_DIR = '/root/.openclaw/workspace/github-pages'


def load_original_character_frames():
    """Load d615a1e frames and extract per-frame character color grids."""
    result = subprocess.run(
        ['git', 'show', 'd615a1e:frames.js'],
        capture_output=True, text=True, cwd=WORK_DIR
    )
    src = result.stdout
    
    # Extract FRAME_DATA
    start = src.index('FRAME_DATA = "') + len('FRAME_DATA = "')
    end = src.index('"', start)
    b64data = src[start:end]
    
    compressed = base64.b64decode(b64data)
    raw = zlib.decompress(compressed)
    
    frames = []
    offset = 0
    while offset < len(raw):
        rle_len = (raw[offset] << 8) | raw[offset + 1]
        offset += 2
        grid = np.zeros(TARGET_H * TARGET_W, dtype=np.uint8)
        idx = 0
        for i in range(rle_len):
            count = raw[offset + i * 2]
            val = raw[offset + i * 2 + 1]
            for _ in range(count):
                if idx < TARGET_H * TARGET_W:
                    grid[idx] = val
                    idx += 1
        frames.append(grid.reshape(TARGET_H, TARGET_W))
        offset += rle_len * 2
    
    return frames


def extract_video_frames():
    """Extract frames from video at target FPS, resize to target resolution."""
    cap = cv2.VideoCapture(os.path.join(WORK_DIR, 'badapple.mp4'))
    frames = []
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    step = VIDEO_FPS / TARGET_FPS
    idx = 0.0
    while int(idx) < video_frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        idx += step
    
    cap.release()
    return frames


def build_character_templates(original_frames):
    """
    Build silhouette templates for each character from the original 900 frames.
    Returns dict: color_id -> list of (frame_idx, silhouette_mask) pairs
    """
    templates = {}  # color_id -> [(frame_idx, mask)]
    
    for i, grid in enumerate(original_frames):
        # Find which character colors are in this frame
        unique_colors = set(grid.flatten())
        char_colors = [c for c in unique_colors if c >= 2]
        
        if not char_colors:
            continue
        
        # For each character color, extract its silhouette mask
        for color_id in char_colors:
            mask = (grid == color_id).astype(np.uint8)
            if mask.sum() < 10:  # Skip tiny fragments
                continue
            if color_id not in templates:
                templates[color_id] = []
            templates[color_id].append((i, mask))
    
    return templates


def compute_silhouette_features(mask):
    """Compute shape features for a silhouette mask."""
    # Hu moments (rotation/scale invariant)
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform for better comparison
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    # Bounding box aspect ratio
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        aspect = w / max(h, 1)
        area = cv2.contourArea(contours[0])
        extent = area / max(w * h, 1)
    else:
        aspect, extent = 0, 0
    
    return np.concatenate([hu, [aspect, extent]])


def match_frame_to_character(gray_frame, templates, threshold=0.5):
    """
    Match a grayscale video frame to the best character template.
    Returns the character color ID, or 1 (white) if no good match.
    """
    # Binarize the grayscale frame
    _, binary = cv2.threshold(gray_frame, 30, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    
    if binary.sum() < 100:
        return 1  # Mostly black frame
    
    frame_features = compute_silhouette_features(binary)
    
    best_color = 1
    best_score = float('inf')
    
    for color_id, template_list in templates.items():
        for tmpl_idx, tmpl_mask in template_list:
            # Quick size check
            tmpl_size = tmpl_mask.sum()
            frame_size = binary.sum() / 255
            size_ratio = min(tmpl_size, frame_size) / max(tmpl_size, frame_size)
            
            if size_ratio < 0.3:
                continue  # Too different in size
            
            # Compute features for template
            tmpl_features = compute_silhouette_features(tmpl_mask)
            
            # Feature distance
            dist = np.linalg.norm(frame_features - tmpl_features)
            
            if dist < best_score:
                best_score = dist
                best_color = color_id
    
    if best_score > threshold:
        return 1  # No good match, default to white
    
    return best_color


def build_timing_schedule(original_frames):
    """
    Build a timing-based character schedule from the original 900 frames.
    This is more reliable than per-frame template matching.
    """
    # For each frame, find the dominant character color
    frame_chars = []
    for grid in original_frames:
        unique = set(grid.flatten())
        chars = [c for c in unique if c >= 2]
        if not chars:
            frame_chars.append(1)
        elif len(chars) == 1:
            frame_chars.append(chars[0])
        else:
            # Multiple chars - use the one with most pixels
            counts = {}
            for c in chars:
                counts[c] = (grid == c).sum()
            frame_chars.append(max(counts, key=counts.get))
    
    # Build segments of consistent character
    segments = []
    current = frame_chars[0]
    start = 0
    for i, c in enumerate(frame_chars):
        if c != current:
            segments.append((start, i, current))
            current = c
            start = i
    segments.append((start, len(frame_chars), current))
    
    return segments, frame_chars


def get_character_for_frame(frame_idx, total_frames, segments, frame_chars):
    """
    Determine character color for a frame index.
    Uses the original 900-frame mapping for the first 75s,
    then extrapolates using repetition patterns.
    """
    original_count = 900
    original_duration = original_count / TARGET_FPS  # 75s
    
    if frame_idx < original_count:
        # Direct lookup
        return frame_chars[frame_idx]
    
    # For frames beyond the original, use the song structure
    # Bad Apple structure (approximate):
    # 0-30s: intro (mostly white, then Reimu)
    # 30-75s: verse 1 (Reimu, Marisa, mixed)
    # 75-120s: chorus 1 (more characters)
    # 120-165s: verse 2 
    # 165-210s: chorus 2
    # 210-219s: outro
    
    # Map the original 75s pattern across the full song
    # The verse/chorus structure repeats with variations
    time_in_song = frame_idx / TARGET_FPS
    
    # Use a cyclical mapping based on the song structure
    # Each "verse" is ~45s, each "chorus" is ~45s
    # Map back to the 0-75s range
    cycle_time = time_in_song % original_duration
    lookup_frame = int(cycle_time * TARGET_FPS)
    lookup_frame = min(lookup_frame, original_count - 1)
    
    return frame_chars[lookup_frame]


def generate_frames():
    """Main generation pipeline."""
    print("=== Bad Apple!! Character-Colored Frame Generator ===\n")
    
    # Step 1: Load original character mapping
    print("Step 1: Loading original 900-frame character mapping...")
    original_frames = load_original_character_frames()
    print(f"  Loaded {len(original_frames)} frames")
    
    # Step 2: Build timing schedule
    print("Step 2: Building character timing schedule...")
    segments, frame_chars = build_timing_schedule(original_frames)
    print(f"  {len(segments)} character segments:")
    for start, end, cid in segments[:15]:
        t_start = start / TARGET_FPS
        t_end = end / TARGET_FPS
        print(f"    {t_start:.1f}s-{t_end:.1f}s: color {cid}")
    if len(segments) > 15:
        print(f"    ... and {len(segments)-15} more")
    
    # Step 3: Extract video frames
    print("\nStep 3: Extracting video frames...")
    video_frames = extract_video_frames()
    print(f"  Extracted {len(video_frames)} frames at {TARGET_FPS}fps")
    
    # Step 4: Generate colored frames
    print("\nStep 4: Generating character-colored frames...")
    all_rles = []
    color_usage = {}
    
    for i, gray in enumerate(video_frames):
        if i >= 2629:
            break
        
        # Determine character color
        char_color = get_character_for_frame(i, 2629, segments, frame_chars)
        
        # Binarize the grayscale frame
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Create the color grid
        grid = np.zeros(TARGET_H * TARGET_W, dtype=np.uint8)
        grid[binary.flatten() > 0] = char_color
        
        # RLE encode
        rle = []
        flat = grid
        j = 0
        while j < len(flat):
            val = flat[j]
            count = 1
            while j + count < len(flat) and flat[j + count] == val and count < 255:
                count += 1
            rle.append((count, int(val)))
            j += count
        all_rles.append(rle)
        
        color_usage[char_color] = color_usage.get(char_color, 0) + 1
        
        if i % 500 == 0:
            print(f"  frame {i}/{len(video_frames)}: char_color={char_color}")
    
    print(f"\n  Color usage summary:")
    for cid in sorted(color_usage.keys()):
        pct = color_usage[cid] / len(video_frames) * 100
        print(f"    ID {cid}: {color_usage[cid]} frames ({pct:.1f}%)")
    
    # Step 5: Pack and compress
    print("\nStep 5: Packing and compressing...")
    data = bytearray()
    for rle in all_rles:
        data.extend(struct.pack('>H', len(rle)))
        for count, val in rle:
            data.extend(struct.pack('BB', count, val))
    
    compressed = zlib.compress(bytes(data), 9)
    b64data = base64.b64encode(compressed).decode('ascii')
    print(f"  Raw: {len(data)} bytes")
    print(f"  Compressed: {len(compressed)} bytes")
    print(f"  Base64: {len(b64data)} chars")
    
    # Step 6: Write frames.js
    colors_json = '{'
    for k in sorted(CHAR_COLORS.keys()):
        v = CHAR_COLORS[k]
        colors_json += f'"{k}": [{v[0]}, {v[1]}, {v[2]}], '
    colors_json = colors_json.rstrip(', ') + '}'
    
    output = f"""// Bad Apple!! — Character-mapped frame data
// {len(all_rles)} frames, {TARGET_W}x{TARGET_H} @ {TARGET_FPS}fps
// Character color palette
const CHAR_COLORS = {colors_json};
const FRAME_COUNT = {len(all_rles)};
const DOT_COLS = {TARGET_W};
const DOT_ROWS = {TARGET_H};
const FPS = {TARGET_FPS};
const FRAME_DATA = "{b64data}";"""
    
    out_path = os.path.join(WORK_DIR, 'frames.js')
    with open(out_path, 'w') as f:
        f.write(output)
    
    print(f"\n✅ Generated {out_path}")
    print(f"   {len(all_rles)} frames, {TARGET_W}x{TARGET_H} @ {TARGET_FPS}fps")
    print(f"   File size: {len(output)} chars")


if __name__ == '__main__':
    generate_frames()
