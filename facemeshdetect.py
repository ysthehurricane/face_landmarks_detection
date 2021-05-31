import cv2
import mediapipe as mp
import time

class faceMeshDetection:
    
    def __init__(self, static_mode=False, maxfaces=2, detection_confident= 0.5, tracking_confident=0.5):
        self.static_mode = static_mode
        self.maxfaces = maxfaces
        self.detection_confident = detection_confident
        self.tracking_confident = tracking_confident
        
        self.mpdraw = mp.solutions.drawing_utils
        self.mpmeshes = mp.solutions.face_mesh
        self.facemesh = self.mpmeshes.FaceMesh(self.static_mode, self.maxfaces,\
                                               self.detection_confident, self.tracking_confident) 
        self.drawspec = self.mpdraw.DrawingSpec(thickness=1, circle_radius=2)

        
    def findfacemeshes(self, frame, draw_landmark=True):
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.facemesh.process(img)
        
        faces = []
        
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw_landmark:
                    self.mpdraw.draw_landmarks(frame, facelms, self.mpmeshes.FACE_CONNECTIONS, \
                                           self.drawspec, self.drawspec)
                face = []
                
                for idx,lm in enumerate(facelms.landmark):
                    h,w,c = frame.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    #cv2.putText(frame, str(idx), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,255,0), 1)
                    face.append([cx,cy])

            faces.append(face)

        return frame, faces
    