import vidstab
from vidstab import VidStab
import numpy as np
import cv2

def preprocess (in_path, out_path):
    #stabilize video
    #stabilizer = VidStab()
    #stabilizer.stabilize(input_path=in_path, output_path=out_path)

    #Resize
    print('Resizing {}'.format(out_path))
    cap = cv2.VideoCapture(in_path)
        
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError
    
    result = cv2.VideoWriter(out_path, 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            15, (720, 480))
    
    while cap.isOpened():
            
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        #resize the video to 256 x 256 (as Olivier did)
        resize = cv2.resize(img, (720, 480)) #, interpolation =cv2.INTER_AREA)
        #save the resized frame
        result.write(resize)

        # Closing the video by Escape button
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    #cap.release()
    #result.release()

def compute_angle (a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    # Calculate the Angle
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle < 0:
        angle += 360
        
    return angle 

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

""" def polynomial_fitting (missing_dict)
    MISSING_RANGE = 30 # the collecting range.
    for key, val in missing_dict.items():
        cnt += 1; print(end=f"\rFitting missing points with Polynomial Fitting: {cnt} / {total_missing}")
        for pt in val:
            m_x, m_y = [], [] # collect all x and y values for y-input for polynomial fitting
            idx_list_x, idx_list_y = [], [] # collect all frame numbers as x-input for polynomial fitting
            for idx in range(1, MISSING_RANGE+1): 
                # check key still in-bound
                if (key - idx) >= 0 and keypoints[key-idx][pt][0] != 0.0:
                    m_x.append(keypoints[key-idx][pt][0])
                    idx_list_x.append(key-idx)

                if (key - idx) >= 0 and keypoints[key-idx][pt][1] != 0.0:
                    m_y.append(keypoints[key-idx][pt][1])
                    idx_list_y.append(key-idx)
                    
                if (key + idx) < total_frames and keypoints[key+idx][pt][0] != 0.0:
                    m_x.append(keypoints[key+idx][pt][0])
                    idx_list_x.append(key+idx)

                if (key + idx) < total_frames and keypoints[key+idx][pt][1] != 0.0:
                    m_y.append(keypoints[key+idx][pt][1])
                    idx_list_y.append(key+idx)
            
            # fill missing points with polynomial fitting
            poly_x = np.polyfit(idx_list_x, m_x, 2)
            poly_y = np.polyfit(idx_list_y, m_y, 2)
            # forming the polynomial function and given the "key" as the x-input
            fit_eq_x = poly_x[0] * np.square(key) + poly_x[1] * key + poly_x[2]
            fit_eq_y = poly_y[0] * np.square(key) + poly_y[1] * key + poly_y[2]
            # assign to the missing points
            keypoints[key][pt][0] = fit_eq_x
            keypoints[key][pt][1] = fit_eq_y
 """
# =========================================================================================
# code from https://github.com/AssemblyAI-Examples/mediapipe-python.git
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import mediapipe as mp

def scale_axes(ax):
    # Scale axes properly
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])


def plot_data(data, ax, pose_connections, rotate=True):
    if rotate:
        ax.scatter(data[0, :], data[2, :], -data[1, :])

        for i in pose_connections:
            ax.plot3D([data[0, i[0]], data[0, i[1]]],
                      [data[2, i[0]], data[2, i[1]]],
                      [-data[1, i[0]], -data[1, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=10, azim=-60)

    else:
        ax.scatter(data[0, :], data[1, :], data[2, :])

        for i in pose_connections:
            ax.plot3D([data[0, i[0]], data[0, i[1]]],
                      [data[1, i[0]], data[1, i[1]]],
                      [data[2, i[0]], data[2, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=-90, azim=-90)


def rotate_and_save(figure, axis, filename, save=False):
    def init():
        return figure,

    def animate(i):
        axis.view_init(elev=10., azim=i)
        return figure,

    # Animate
    anim = animation.FuncAnimation(figure, animate, init_func=init,
                                   frames=800, interval=20, blit=True)
    plt.close()

    # Save
    if save:
        anim.save(filename, fps=10, extra_args=['-vcodec', 'libx264'], dpi=300)


def time_animate(data, figure, ax, pose_connections, rotate_data=True, rotate_animation=False):
    frame_data = data[:, :, 0]
    if rotate_data:
        plot = [ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:blue')]

        for i in pose_connections:
            plot.append(ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                                  [-frame_data[2, i[0]], -frame_data[2, i[1]]],
                                  [-frame_data[1, i[0]], -frame_data[1, i[1]]],
                                  color='k', lw=1)[0])

        ax.view_init(elev=10, azim=120)

    else:
        ax.scatter(frame_data[0, :], frame_data[1, :], frame_data[2, :], color='tab:blue')

        for i in pose_connections:
            ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                      [frame_data[1, i[0]], frame_data[1, i[1]]],
                      [frame_data[2, i[0]], frame_data[2, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=-90, azim=-90)

    scale_axes(ax)

    def init():
        return figure,

    def animate(i):
        frame_data = data[:, :, i]

        for idxx in range(len(plot)):
            plot[idxx].remove()

        plot[0] = ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:blue')

        idx = 1
        for pse in pose_connections:
            plot[idx] = ax.plot3D([frame_data[0, pse[0]], frame_data[0, pse[1]]],
                                  [-frame_data[2, pse[0]], -frame_data[2, pse[1]]],
                                  [-frame_data[1, pse[0]], -frame_data[1, pse[1]]],
                                  color='k', lw=1)[0]
            idx += 1

        if rotate_animation:
            ax.view_init(elev=10., azim=120 + (360 / data.shape[-1]) * i)

        return figure,

    # Animate
    anim = animation.FuncAnimation(figure, animate, init_func=init,
                                   frames=800, interval=10, blit=True)

    plt.close()

    return anim