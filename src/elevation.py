"""
Elevation fetching utilities

Fetches elevation data from Swiss geo.admin.ch height service.
"""

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from threading import Lock

# Rate limiting: 5 requests per second max
_last_request_time = 0
_request_lock = Lock()


def _fetch_single_elevation(coord: Tuple[float, float]) -> float:
    """
    Fetch elevation for a single coordinate with rate limiting.

    Args:
        coord: Tuple of (x, y) coordinates

    Returns:
        Elevation value or None on failure
    """
    global _last_request_time

    x, y = coord
    url = "https://api3.geo.admin.ch/rest/services/height"

    # Rate limiting: max 5 requests per second
    with _request_lock:
        current_time = time.time()
        time_since_last = current_time - _last_request_time
        if time_since_last < 0.2:  # 0.2 seconds = 5 req/s
            time.sleep(0.2 - time_since_last)
        _last_request_time = time.time()

    try:
        res = requests.get(url, params={"easting": x, "northing": y, "sr": "2056"}, timeout=10)
        res.raise_for_status()
        h = float(res.json()["height"])
        return h
    except Exception:
        return None


def fetch_elevation_batch(coords: List[Tuple[float, float]], batch_size: int = 50, 
                          delay: float = 0.1, max_workers: int = 15) -> List[float]:
    """
    Fetch elevations for a list of coordinates via geo.admin.ch REST height service.
    Uses concurrent requests for faster processing.
    
    Args:
        coords: List of (x, y) coordinate tuples
        batch_size: Progress reporting interval (deprecated, kept for compatibility)
        delay: Delay between batches (deprecated, kept for compatibility)
        max_workers: Number of concurrent workers (default: 15)
        
    Returns:
        List of elevations in same order as coords
    """
    total = len(coords)
    elevations = [None] * total
    failed_count = 0
    
    print(f"Fetching elevations for {total} points (concurrent, {max_workers} workers)...")
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_fetch_single_elevation, coord): i 
            for i, coord in enumerate(coords)
        }
        
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                elevation = future.result()
                if elevation is not None:
                    elevations[index] = elevation
                else:
                    failed_count += 1
                    elevations[index] = None
            except Exception:
                failed_count += 1
                elevations[index] = None
            
            completed += 1
            if completed % batch_size == 0:
                pct = completed / total * 100
                print(f"  Progress: {completed}/{total} ({pct:.1f}%)")
    
    # Post-processing: replace None values with average of successful elevations
    successful_elevations = [e for e in elevations if e is not None]
    if successful_elevations:
        fallback_value = sum(successful_elevations) / len(successful_elevations)
    else:
        # If all failed, use a reasonable Swiss elevation (500m typical lowland)
        fallback_value = 500.0
        print("  Warning: All elevation fetches failed, using fallback of 500.0m")

    for i in range(total):
        if elevations[i] is None:
            elevations[i] = fallback_value
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} points failed to fetch elevation")
    
    return elevations

