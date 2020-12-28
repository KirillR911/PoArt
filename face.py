import PIL
import cv2
import face
import numpy as np
import io 
from math import acos, pi, sin, cos
import face_alignment
from utils import vis_landmark_on_img, procrustes, overlay_image_alpha, get_rect,rotate_around_point
class Face():
    def __init__(self, params, fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')):
        if params['fromPath']:
            self.imgcv2 = cv2.imread(params['path'])
        elif params['fromBytes']:
            self.imgpil = PIL.Image.open(io.BytesIO(params['bytes'])).convert("RGB")
            self.imgcv2 = np.array(self.imgpil) [:, :, ::-1].copy()
        self.imgpil  = PIL.Image.fromarray(self.imgcv2 [:, :, ::-1])
        self.lms = fa.get_landmarks(self.imgcv2)[0] 
        self.paired_imgcv2 = cv2.imread(params['style'])
        self.paired_impil = PIL.Image.open(params['style'])
        assert self.lms.shape == (68, 2)
        vis_landmark_on_img(self.lms.copy(), name="face.jpg")
        vis_landmark_on_img(params['styleLms'].copy())
        _, new, tform = procrustes(self.lms,params['styleLms'].copy())
        vis_landmark_on_img(new.copy(), name="aligned.jpg")
        print(tform)
        patern1 = [10, 52,8,14]
        rect = cv2.boundingRect(params['styleLms'][patern1])
        x,y,w,h = rect
        croped = self.paired_imgcv2[y:y+int(h), x:x+int(w)].copy()
        croped =  cv2.resize(croped, (int(w*tform['scale']),int(h*tform['scale'])))
        cv2.imwrite("croped.jpg",croped)
        angle = acos(tform['rotation'][0][0]) 
        angle *= (sin(angle))/ (tform['rotation'][1][0])
        pilcroped = PIL.Image.fromarray(croped[:, :, ::-1])
        pilcroped = pilcroped.convert("RGBA")
        pilcroped.save("tmp1.png")
        pilcroped = pilcroped.rotate(angle/pi * 180, expand = 1)
        pilcroped.save("tmp.png")
        target_rec = cv2.boundingRect(new[patern1])
        rec_points = [
            (target_rec[0], target_rec[1]),
            (target_rec[0]+target_rec[2], target_rec[1]),
            (target_rec[0]+target_rec[2],target_rec[1]+target_rec[3]),
            (target_rec[0],target_rec[1]+target_rec[3]),
            (target_rec[0], target_rec[1])
            ]
        # rect_poits = [target_rec[0], target_rec[1],target_rec[2],target_rec[3],target_rec[0]]
        final_rec = [[*rotate_around_point(i, radians = angle, origin = (target_rec[0]+target_rec[2]/2, target_rec[1]+target_rec[2]/2))] for i in rec_points ]
        print(final_rec)
        vis_landmark_on_img(self.lms, name = "overla.jpg", img= self.imgcv2.copy(), normal=False, rect = [
                                                                                                    target_rec[0],
                                                                                                    target_rec[1],
                                                                                                    target_rec[0]+target_rec[2],
                                                                                                    target_rec[1]+target_rec[3], 
                                                                                                    
                                                                                                    ]
                            )
        vis_landmark_on_img(new, name = "overla2.jpg", img=self.imgcv2.copy(), normal=False,poly = final_rec)
        print(angle/pi * 180)

        # self.imgpil = PIL.Image.open("overla2.jpg")
        # self.imgpil.paste(pilcroped,(int(final_rec[0][0]), int(final_rec[0][1]-w*tform['scale']*cos(angle)/2)), pilcroped)
        self.imgpil.paste(pilcroped,(int(final_rec[0][0]), int(final_rec[0][1]-w*tform['scale']*abs(sin(angle)))), pilcroped)
        # self.imgpil.paste(pilcroped,(int(final_rec[0][0]), int(final_rec[0][1]+w*tform['scale']*sin(angle))), pilcroped)


        # self.imgpil.paste(pilcroped,(int(final_rec[0][0]), int(final_rec[0][1]-h/2)), pilcroped)

        self.imgpil.save("warped.jpg")
        # M = cv2.getRotationMatrix2D((w/2,h/2),angle/pi * 180,1)
        # new_image = cv2.warpAffine(croped, M, dsize=(w, h))
        # new_image = cv2.resize(new_image, (int(w*tform['scale']),int(h*tform['scale'])))
        # ovrd = self.imgcv2.copy()
        # print(ovrd.shape)
        # print(new_image.shape)
        # mask = np.ones((new_image.shape[0], new_image.shape[1]))
        # overlay_image_alpha(ovrd, new_image, (50,50), mask)
        # cv2.imwrite("warped.jpg",ovrd)
        
        
        
        
        
        
        