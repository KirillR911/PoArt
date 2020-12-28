import os, glob, time, sys
import numpy as np
from PIL import Image
import cv2
from sklearn.preprocessing import scale


def intert_func(input_mat, src_fps=25, trg_fps=62.5):
    xp = list(np.arange(0, input_mat.shape[0], 1))
    interp_xp = list(np.arange(0, input_mat.shape[0], src_fps/trg_fps))
    interp_mat = np.zeros(shape=(len(interp_xp), input_mat.shape[1]))
    for j in range(input_mat.shape[1]):
        interp_mat[:, j] = np.interp(interp_xp, xp, input_mat[:, j])
    return interp_mat

lms = np.loadtxt("trump.txt").reshape((-1,68,3))
lms[:,:,:] -= np.min(lms)
lms *= (256 / np.max(lms))
lms[:,:,0] += 40
center_point = np.mean(lms[0], axis = 0)
lms[:,:,:3] -= center_point
print(center_point)
nlms = lms[:,:,:2]
lms *= 1.3
lms[:,:,:3] += center_point
lms *= (256 / np.max(lms))
# for idx in range(len(nlms)):
#     nlms[idx] = scale(nlms[idx], 1.2)
# lms += 256
# tmp_lms = np.load("5/LMS.npy")
# lms[:,:,0] -= -7.888240e-01
# lms[:,:,1] -= -1.400800e-02
# lms[:,:,2] -= -7.389500e-02
# print(lms[0])
# lms /= np.max(lms)
# lms *=256
# lms[:,:,0] -= 70
# lms[:,:,1] -= 70

# lms[:,:,0] += -2.08071411e+02
# lms[:,:,1] += -1.51859772e+02
# lms[:,:,2] += -2.09230011e+02
# lms *= 1.5
# lms += 100
# lms[:,:,0]+=20
np.save("target.npy", nlms[:,:,:2])

print(nlms[0])
# print(tmp_lms[0])
# lms = tmp_lms.reshape((-1,204))
# lms = intert_func(lms)
# lms = lms.reshape((-1,68,3))
# lms[:,49:54, 1] += 1.           # thinner upper lip
# lms[:,55:60, 1] -= 1.           # thinner lower lip
# lms[:,[37,38,43,44], 1] -=3.    # larger eyes
# lms[:,[40,41,46,47], 1] +=3. 
# lms[:,[48,49,59,60,61,67], 0] += 3.5
# lms[:,[63,64,65,53,55,54], 0] -= 3.5
# lms[:, 48:, 0] = (lms[:, 48:, 0] - np.mean(lms[:, 48:, 0])) * 1.05 + np.mean(lms[:, 48:, 0])
def vis_landmark_on_img(shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''
    img = np.ones(shape=(256, 256, 3)) * 255
    # shape += 256
    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (int(shape[i, 0]), int(shape[i, 1])), (int(shape[i + 1, 0]), int(shape[i + 1, 1])), color, lineWidth)
        if (loop):
            cv2.line(img, (int(shape[idx_list[0], 0]), int(shape[idx_list[0], 1])),
                     (int(shape[idx_list[-1] + 1, 0]), int(shape[idx_list[-1] + 1, 1])), color, lineWidth)

    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))

    return img

frames_l= []
for rex in nlms:
    data = vis_landmark_on_img(rex)
    im = Image.fromarray((data).astype(np.uint8))
    frames_l.append(im)
    
frames_l[0].save('moving_text_t.gif', format='GIF',
               append_images=frames_l[1:], save_all=True, duration=5.339138321995465, loop=0)