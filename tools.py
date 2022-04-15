import dlib
import os
import numpy as np

file_path = os.path.dirname(__file__)

class face_detector:
    def __init__(self,opt="hog"):
        opt_dict = {
                    "hog" : dlib.get_frontal_face_detector(),
                    "cnn" : dlib.cnn_face_detection_model_v1(os.path.join(file_path,'weight/mmod_human_face_detector.dat')),
                    }
        self.model = opt_dict[opt]

    def __call__(self,img, n_upsample=1, ret="rect"):
        '''
        img : input image
        n_upsample : number of time to upsample
        ret : return format
                "bb" : for bounding box ( (left,top) , (right,bottom) )
                "rect" : dlib rect format  
        '''
        faces = self.model(img, n_upsample)
        if ret == "bb":
            return [rect2bb(face) for face in faces]
        elif ret == "rect":
            return faces

class face_features:
    def __init__(self, landmark="large"):
        landmark_dict = {
            "small" : dlib.shape_predictor(os.path.join(file_path,'weight/shape_predictor_5_face_landmarks.dat')),
            "large" : dlib.shape_predictor(os.path.join(file_path,'weight/shape_predictor_68_face_landmarks.dat')),
        }
        self.landmark = landmark_dict[landmark]
        self.encoder = dlib.face_recognition_model_v1(os.path.join(file_path,'weight/dlib_face_recognition_resnet_model_v1.dat'))
        
    def __call__(self, img, face_locations):
        landmarks = self.get_landmark(img,face_locations)
        encode = np.array([self.encoder.compute_face_descriptor(img, landmark) for landmark in landmarks])
        return encode
    def get_landmark(self, img, face_locations):
        return np.array([self.landmark(img, face) for face in face_locations])

    def from_img(self,img, faceDetector=None):
        if faceDetector is None:
            faceDetector = face_detector()
        face_locations = faceDetector(img)
        return self.__call__(img,face_locations)


def rect2bb(rect):
    return (rect.left(),rect.top()) , (rect.right(), rect.bottom())

def face_compare(enc1, enc2, distance = "cosine", threshold = None, return_dist = False):
    '''
        distance : distance function "cosine", "l1", "l2"
    '''
    threshold_dict = {'cosine': 0.07, 'l1': 0.6, 'l2': 0.4}
    dist_dict = {
        "cosine" : cosine_dist,
        "l1" : l1_dist,
        "l2" : l2_dist, 
    }
    dist = dist_dict[distance](enc1, enc2)
    if threshold is None:
        threshold = threshold_dict[distance]
    if return_dist:
        return (dist <= threshold).item(), dist.item()
    else: 
        return (dist <= threshold).item()


def cosine_dist(a, b):
    return 1 - ((a@b.T)/(np.linalg.norm(a)*np.linalg.norm(b)))

def l1_dist(a, b):
    return np.linalg.norm(a-b, ord=1, axis=1)

def l2_dist(a, b):
    return np.linalg.norm(a-b, ord=2, axis=1)