from face import Face
import face_alignment

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
params = {
    'fromPath' : True,
    'fromBytes' : False, 
    'path' : "./m.jpg",
    'style' : "style/climt.jpg",
    'styleLms' : fa.get_landmarks("style/climt.jpg")[0]
}
f = Face(params, fa)
print(f.lms)

