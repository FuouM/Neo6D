import os
import math
from math import cos, sin

import numpy as np
import torch
import scipy.io as sio
import cv2

# batch*n
def normalize_vector(v, use_gpu=True, gpu_id=0):
    # Calculate the magnitude of each vector in the batch
    v_mag = torch.sqrt(torch.sum(v**2, dim=1))  # shape: (batch_size,)
    
    # Add a small constant to prevent division by zero
    small_value = 1e-8
    if use_gpu:
        small_value_tensor = torch.FloatTensor([small_value]).cuda(gpu_id)
    else:
        small_value_tensor = torch.FloatTensor([small_value])
    v_mag = torch.max(v_mag, small_value_tensor)

    # Normalize each vector by its magnitude
    v_norm = v / v_mag.view(-1, 1).expand_as(v)

    return v_norm
    
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    
    # Compute the components of the cross product
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True, gpu_id=0):
    # Split the input tensor into x and y components
    x_raw = poses[:,0:3]  # batch*3
    y_raw = poses[:,3:6]  # batch*3
    
    # Normalize the x and z components
    x = normalize_vector(x_raw, use_gpu, gpu_id=gpu_id)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z, use_gpu, gpu_id=gpu_id)  # batch*3
    
    # Calculate the y component
    y = cross_product(z, x)  # batch*3
    
    # Reshape the components into a 3x3 rotation matrix
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    
    return matrix

def compute_euler_angles_from_rotation_matrices(rotation_matrices,
                                                use_gpu=True, gpu_id=0):
    # Get the batch size and the rotation matrices
    batch_size = rotation_matrices.shape[0]
    R = rotation_matrices
    
    # Calculate the sin and cosine of the pitch angle
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
    
    # Check if the pitch angle is close to 90 degrees
    singular = sy < 1e-6
    singular = singular.float()
    
    # Calculate the Euler angles using the rotation matrix
    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    
    # Calculate the alternative Euler angles for the special case when pitch angle is close to 90 degrees
    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0
        
    # Combine the Euler angles and the alternative Euler angles
    out_euler = torch.zeros(batch_size, 3)
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular
    
    # Move the output tensor to the GPU, if requested
    if use_gpu:
        out_euler = out_euler.cuda(gpu_id)
        
    # Return the Euler angles
    return out_euler

def write_data_to_file(video_path, frame_time, tdx, tdy, yaw, pitch, roll, size):
    # create a file name based on the video file name
    file_name = os.path.splitext(os.path.basename(video_path))[0] + '.csv'
    
    with open(file_name, 'a') as f:
        # write the values to the file
        f.write(f"{frame_time},{tdx},{tdy},{yaw[0]},{pitch[0]},{roll[0]},{size}\n")

def append_line_to_top(file_path, new_line):
    with open(file_path, 'r') as f:
        content = f.read()

    with open(file_path, 'w') as f:
        f.write(new_line + '\n' + content)


def remove_empty_newline(file_path):
    with open(file_path, 'rb+') as file:
        file.seek(-1, os.SEEK_END)
        last_char = file.read(1)
        if last_char == b'\n':
            file.seek(-1, os.SEEK_END)
            file.truncate()
