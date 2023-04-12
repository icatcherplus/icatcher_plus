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
        pass

    def get_ref_image(self, img_path):
        """
        If the user passes a reference image path to the command line, it will be grabbed from here
        param img_path: Path to the reference image to be used by face recognition
        """
        img = face_recognition.load_image_file(img_path)
        self.known_faces.append(face_recognition.face_encodings(img)[0])

    def generate_ref_image(self, bbox, frame):
        """
        Save a location as a reference image

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
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(
                self.known_faces, face_encoding, tolerance=0.10
            )

            if match[0]:
                return face_locations[0]
            else:
                return None