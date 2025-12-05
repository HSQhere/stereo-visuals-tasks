#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

def setting_root():
    # setting data route
    data_folder_left = "data_road/training/image_2/"
    data_folder_right = "data_road/training_right/image_3/"
    data_folder_calib = "data_road/training/calib/"
    prefix = ['uu', 'uum', 'um']
    index_length = 6

    # choose image
    index_num = 0
    prefix_idx = 2  # test um_000001.png
    fname = prefix[prefix_idx] + '_' + str(index_num).zfill(index_length)
    print(f"process: {fname}")
    img_fname = fname + '.png'
    calib_fname = fname + '.txt'

    # read it
    img_left_color = cv2.imread(data_folder_left + img_fname)
    img_right_color = cv2.imread(data_folder_right + img_fname)
    calib_file = data_folder_calib + calib_fname
    return img_left_color,img_right_color,calib_file,fname

def show_origin(img_left_color, img_right_color, img_left_bw, img_right_bw):
    # display original image
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB))
    plt.title('Left Image')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(img_right_color, cv2.COLOR_BGR2RGB))
    plt.title('Right Image')

    plt.subplot(1, 4, 3)
    plt.imshow(img_right_bw, cmap='gray')
    plt.title('Right Grayscale')

    plt.subplot(1, 4, 4)
    plt.imshow(img_left_bw, cmap='gray')
    plt.title('left Grayscale')

    plt.tight_layout()
    plt.show()

def smart_sampling(out_points, out_colors, target_points):

    total_points = len(out_points)

    if total_points <= target_points:
        return out_points, out_colors

    print(f"sampling: {total_points:,} → {target_points:,} points")

    # sampling by depth
    depths = out_points[:, 2]
    depth_bins = np.linspace(depths.min(), depths.max(), 10)
    sampled_indices = []
    for i in range(len(depth_bins)-1):
        mask = (depths >= depth_bins[i]) & (depths < depth_bins[i+1])
        bin_points = np.where(mask)[0]
        if len(bin_points) > 0:
            n_sample = min(len(bin_points), target_points // 10)
            sampled_indices.extend(np.random.choice(bin_points, n_sample, replace=False))
    sample_idx = np.array(sampled_indices)

    return out_points[sample_idx], out_colors[sample_idx]

def visualize_3d_points(out_points, out_colors, target_points):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # stratified sampling
    points_sample, colors_sample = smart_sampling(out_points, out_colors, target_points)

    # draw 3D scattering map
    scatter = ax.scatter3D(
        points_sample[:, 0],  # X
        points_sample[:, 2],  # Z
        points_sample[:, 1],  # Y
        c=colors_sample/255.0,  # colour
        s=3,  # point
        alpha=0.7,  # transparency
        depthshade=True  # deep shadow
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title('3D Point Cloud')
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_plane(out_points,out_colors,target_points):
    # visualize 3D
    points_sampled, color_sampled = smart_sampling(out_points,out_colors,target_points)
    fig = plt.figure(figsize=(15, 5))

    # X-Z plane
    ax1 = fig.add_subplot(1, 3, 1)
    scatter1 = ax1.scatter(points_sampled[:, 0], points_sampled[:, 2], c=points_sampled[:, 2],
                           cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('3D Points (X-Z plane)')
    plt.colorbar(scatter1, ax=ax1)

    # X-Y plane
    ax2 = fig.add_subplot(1, 3, 2)
    scatter2 = ax2.scatter(points_sampled[:, 0], points_sampled[:, 1], c=points_sampled[:, 2],
                           cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('3D Points (X-Y plane)')
    plt.colorbar(scatter2, ax=ax2)

    # Y-Z plane
    ax3 = fig.add_subplot(1, 3, 3)
    scatter3 = ax3.scatter(points_sampled[:, 1], points_sampled[:, 2], c=points_sampled[:, 2],
                           cmap='viridis', s=1, alpha=0.6)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title('3D Points (Y-Z plane)')
    plt.colorbar(scatter3, ax=ax3)
    plt.tight_layout()
    plt.show()

def BM_disparity(img_left_bw, img_right_bw):
    # BM
    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)
    disparity = stereo.compute(img_left_bw, img_right_bw)
    return disparity

def SGBM_disparity(img_left_bw, img_right_bw):
    # create SGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=-16,
        numDisparities=112,
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=5,
        uniquenessRatio=12,
        speckleWindowSize=150,
        speckleRange=45,
        preFilterCap =63,
        mode = cv2.StereoSGBM_MODE_SGBM_3WAY,
    )

    # cal disparity
    disparity = stereo.compute(img_left_bw, img_right_bw)

    return disparity

def show_disparity(img):
    plt.figure(figsize=(10, 5))

    # display origin gray
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Gray Scale')
    plt.colorbar()

    # CMRmap_r
    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap='CMRmap_r')
    plt.title('CMRmap_r Color Map')
    plt.colorbar()

    # Jet
    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='jet')
    plt.title('Jet Color Map')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def capture_cam(calib_file):
    # read calibration file
    matrix_type_1 = 'P2'
    matrix_type_2 = 'P3'

    with open(calib_file, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:2] == matrix_type_1:
                calib_matrix_1 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)
            elif line[:2] == matrix_type_2:
                calib_matrix_2 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)

    print(" P2:")
    print(calib_matrix_1)
    print(" P3:")
    print(calib_matrix_2)

    # extract camara parameters
    cam1 = calib_matrix_1[:, :3]  # left image - P2
    cam2 = calib_matrix_2[:, :3]  # right image - P3

    # calculate baseline direction
    baseline = -calib_matrix_2[0, 3] / calib_matrix_2[0, 0]
    print(f"baseline direction: {baseline}")

    return cam1, cam2, baseline

def get_Q(baseline, cam1, cam2, img_left_color):
    Tmat = np.array([baseline, 0., 0.])

    rev_proj_matrix = np.zeros((4, 4))

    cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                      distCoeffs1=0, distCoeffs2=0,
                      imageSize=img_left_color.shape[:2],
                      R=np.identity(3), T=Tmat,
                      R1=None, R2=None,
                      P1=None, P2=None, Q=rev_proj_matrix)

    print("reprojection matrix Q:")
    print(rev_proj_matrix)
    return rev_proj_matrix

def optimize(img, rev_proj_matrix, img_left_color):
    # reflect to 3D
    points = cv2.reprojectImageTo3D(img, rev_proj_matrix)

    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    points = np.dot(points, reflect_matrix)

    # extract colors from image
    colors = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB)

    # filter by min disparity
    mask = img > img.min()
    out_points = points[mask]
    out_colors = colors[mask]

    # filter by dimension
    idx = np.fabs(out_points[:, 0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    return out_points, out_colors, reflect_matrix, mask, idx

def reflect_check(out_points, reflect_matrix, cam2, img_left_color, img_right_color, mask, idx):
    reflected_pts = np.matmul(out_points, reflect_matrix)
    projected_img, _ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]),
                                         cam2[:3, :3], np.array([0., 0., 0., 0.]))
    projected_img = projected_img.reshape(-1, 2)

    blank_img = np.zeros(img_left_color.shape, 'uint8')
    img_colors = img_right_color[mask][idx].reshape(-1, 3)

    for i, pt in enumerate(projected_img):
        pt_x = int(pt[0])
        pt_y = int(pt[1])
        if pt_x > 0 and pt_y > 0:
            # use the BGR format to match the original image type
            col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
            cv2.circle(blank_img, (pt_x, pt_y), 1, col)

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB))
    plt.title('3D Points check')
    plt.show()

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

if __name__ == "__main__":
    # return value
    img_left_color, img_right_color, calib_file, fname = setting_root()

    # transform gray and blur
    img_left_bw = cv2.blur(cv2.cvtColor(img_left_color, cv2.COLOR_RGB2GRAY), (5, 5))
    img_right_bw = cv2.blur(cv2.cvtColor(img_right_color, cv2.COLOR_RGB2GRAY), (5, 5))
    show_origin(img_left_color, img_right_color, img_left_bw, img_right_bw)
    # choose stereo matching algorithm
    print("\nchoose stereo matching algorithm:")
    print("1. BM (Block Matching) ")
    print("2. SGBM (Semi-Global Block Matching)")

    choice = input("choose algorithm (1/2, default:1): ").strip()

    if choice == "2":
        print("use SGBM ...")
        img = SGBM_disparity(img_left_bw, img_right_bw).copy()
        output_ply = "out_sgbm_"
    else:
        print("use BM ...")
        img= BM_disparity(img_left_bw, img_right_bw).copy()
        output_ply = "out_bm_"

    show_disparity(img)

    cam1, cam2, baseline = capture_cam(calib_file)

    rev_proj_matrix = get_Q(baseline, cam1, cam2, img_left_color)

    out_points, out_colors, reflect_matrix, mask, idx = optimize(img, rev_proj_matrix, img_left_color)

    write_ply(output_ply+fname+'.ply', out_points, out_colors)
    print(f'{output_ply}{fname}.ply saved')

    visualize_3d_points(out_points, out_colors, 300000)

    visualize_plane(out_points, out_colors, 20000)

    reflect_check(out_points, reflect_matrix, cam2, img_left_color, img_right_color, mask, idx)
