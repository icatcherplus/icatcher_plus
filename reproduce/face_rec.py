import numpy as np
import face_recognition

class FaceRec:
    """
    Encapsulates entity recognition and all data associated with it
    Should be used in the following logic:
        if it passes age detection, no reference image, and has 100% confidence, set reference image for current video
        if number of faces in changed from this frame to last, call facerec check
    """

    def __init__(self):
        self.ref_img_path = None
        self.known_faces = list()

    def convert_bounding_boxes(self, bboxes):
        """
        Converts pipeline bounding box for use in face recognition
        param bbox: bounding boxes in [left, top, width, height] form
        :return: bboxes in [top, right, bottom, left]
        """
        faces = []
        for bbox in bboxes:
            left = bbox[0]
            top = bbox[1]
            w = bbox[2]
            h = bbox[3]
                        
            right = w + left
            bottom = top - h

            faces.append([top, right, bottom, left])
        return faces
        

    def get_ref_image(self, img_path):
        """
        If the user passes a reference image file path to the command line, it will be grabbed from here
        param img_path: Path to the reference image to be used by face recognition
        """
        img = face_recognition.load_image_file(img_path)
        self.known_faces.append(face_recognition.face_encodings(img)[0])

    def generate_ref_image(self, bbox, frame):
        """
        Save a bounded location as a reference image

        param bbox: The bounding box that contains the user's reference image
        param frame: The frame that the bounding box is being used in
        """
        self.known_faces.append(
            face_recognition.face_encodings(frame, known_face_locations=[bbox])[0]
        )

    def facerec_check(self, frame, device="cpu"):
        face_locations = []
        face_encodings = []


        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s), note that tolerance is low
            match = face_recognition.compare_faces(
                self.known_faces, face_encoding, tolerance=0.10
            )

            if match[0]:
                #returns in [top, right, bottom, left] format
                top = face_locations[0][0]
                right = face_locations[0][1]
                bottom = face_locations[0][2]
                left = face_locations[0][3]
                
                h = top - bottom
                w =  right - left
                
                #We want left, top, width, height
                return (top, left, w, h)
            else:
                return None
    
    def select_face(self, bboxes, frame, tolerance=0.10):
        """
        selects a correct face from candidates bbox in frame
        :param bboxes: the bounding boxes of candidates
        :param frame: the frame
        :return: the cropped face and its bbox data
        """
        
        face = None
        #encode new bounding boxes
        for bbox in bboxes:
            face_encodings = face_recognition.face_encodings(frame, known_face_locations=[bbox])[0]
            
            #compare against faces
            matches = face_recognition.compare_faces(
                self.known_faces, face_encodings, tolerance=tolerance
            )
            print(matches)
            if matches[0] == True:
                return bbox

        return None
                
    def select_face_preprocessing(self, bbox, frame):
        
        faces = self.convert_bounding_boxes(bbox)
        face_encodings = face_recognition.face_encodings(frame, known_face_locations=faces)

        matches = face_recognition.compare_faces(self.known_faces, face_encodings)

        for i in range(len(matches)):
            if matches[i] == True:
                selected_face = i
                face = bbox[i]

        crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
                                        # resized_img = cv2.resize(crop_img, (100, 100))
        resized_img = crop_img  # do not lose information in pre-processing step!
        face_box = np.array([face[1], face[1] + face[3], face[0], face[0] + face[2]])
        img_shape = np.array(frame.shape)
        ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                    face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
        face_size = (ratio[1] - ratio[0]) * (ratio[3] - ratio[2])
        face_ver = (ratio[0] + ratio[1]) / 2
        face_hor = (ratio[2] + ratio[3]) / 2
        face_height = ratio[1] - ratio[0]
        face_width = ratio[3] - ratio[2]
        feature_dict = {
            'face_box': face_box,
            'img_shape': img_shape,
            'face_size': face_size,
            'face_ver': face_ver,
            'face_hor': face_hor,
            'face_height': face_height,
            'face_width': face_width
            }
        return selected_face, feature_dict, resized_img