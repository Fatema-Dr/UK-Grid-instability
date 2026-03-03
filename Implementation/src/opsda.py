# src/opsda.py
# Implementation of the Swinging Door Algorithm
# Adapted from a public implementation.

def compress(data, width):
    """
    Compresses a list of (timestamp, value) tuples using the Swinging Door Algorithm.
    """
    if not data:
        return []

    compressed_data = [data[0]]
    start_point_index = 0
    
    for i in range(1, len(data)):
        current_point = data[i]
        pivot_point = data[start_point_index]
        
        # Form a "door" from the pivot point to the current point
        upper_bound_slope = (pivot_point[1] + width - current_point[1]) / (pivot_point[0] - current_point[0]) if pivot_point[0] != current_point[0] else float('inf')
        lower_bound_slope = (pivot_point[1] - width - current_point[1]) / (pivot_point[0] - current_point[0]) if pivot_point[0] != current_point[0] else float('-inf')

        # Check all intermediate points
        for j in range(start_point_index + 1, i):
            intermediate_point = data[j]
            slope = (pivot_point[1] - intermediate_point[1]) / (pivot_point[0] - intermediate_point[0]) if pivot_point[0] != intermediate_point[0] else float('inf')
            
            if slope > upper_bound_slope or slope < lower_bound_slope:
                # Point is outside the door, so we record the previous point and start a new door
                compressed_data.append(data[i-1])
                start_point_index = i - 1
                break
    
    # Add the last point
    compressed_data.append(data[-1])
    
    return compressed_data
