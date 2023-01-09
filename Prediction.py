import cv2
import pickle
import face_recognition


class Predictor:

    @staticmethod
    def predict(img_path, knn_clf=None, model_path=None, threshold=0.6):  # 6 = 40+ , 4 = 60+
        if not (knn_clf or model_path):
            raise Exception("No model supplied")
        # Load the passed KNN model.
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)
        # Get the face location of the passed image.
        img = img_path
        face_box = face_recognition.face_locations(img)
        # return an empty list If no faces are detected in the picture.
        if len(face_box) == 0:
            return []
        # Get faces encodings.
        faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_box)
        # Get best matches for the existing faces.
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
        print(f"{closest_distances = }")
        matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_box))]
        print(f"{matches = }")
        # predict classes and remove the matched points which are not with in the threshold.
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), face_box, matches
                    )]


class CV2:

    @staticmethod
    def set_cv2_props(prediction):
        font = cv2.FONT_HERSHEY_DUPLEX
        for name, (top, right, bottom, left) in prediction:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(frame, name, (left - 10, top - 6), font, 0.8, (255, 255, 255), 1)

        cv2.imshow('Video', frame)


if __name__ == '__main__':

    webcam = cv2.VideoCapture("testing-assets/top-players.jpeg")  # 0 to use webcam
    while True:
        # Iterate till the cam works
        rval = False
        while not rval:
            # Extract the frame from the webcam.
            (rval, frame) = webcam.read()
            if not rval:
                print("Camera is not opened...")
                exit(0)

        frame = cv2.flip(frame, 1)
        frame_copy = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # Sets the image to the appropriate settings of face_recognition.
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        predictions = Predictor.predict(frame_copy, model_path="./model/trained_knn_model.clf")  # add path here
        CV2.set_cv2_props(prediction=predictions)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()
