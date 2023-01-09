import os
import math

import pickle
import numpy as np
import face_recognition
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder


class Trainer:
    def __init__(self):
        self.x = []
        self.y = []
        self.missed_pictures = 0

    def train(self, train_dir, model_dir, neighbours_num=5, algorithm='ball_tree'):
        # Iterate over every directory in train_dir
        people = [person for person in os.listdir(train_dir)]
        for person_dir in people:
            if not os.path.isdir(os.path.join(train_dir, person_dir)):  # training-assets/Human
                continue
            # Iterate of the image of the current directory.
            for img_path in image_files_in_folder(os.path.join(train_dir, person_dir)):
                img = face_recognition.load_image_file(img_path)
                num_of_faces = face_recognition.face_locations(img)
                print("Checking :", img_path)
                if len(num_of_faces) != 1:
                    # If there are no people or too many people in a training image, skip the image.
                    print("Image {} skipped because of recognizing {} faces".format(img_path, len(num_of_faces)))
                    self.missed_pictures += 1
                else:
                    # append the current face to our training set
                    self.x.append(
                        face_recognition.face_encodings(img, known_face_locations=num_of_faces))  # Input features
                    self.y.append(person_dir)  # Outcome

        if neighbours_num is None:
            # Assigning n_neighbors according to Rule of thumb approach.
            neighbours_num = int(round(math.sqrt(len(self.x))))
            print("Assign neighbors: ", neighbours_num)
        # Creating the knn model and fitting it with x,y.
        knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=neighbours_num, algorithm=algorithm,
                                                        weights='distance')
        self.x = Trainer.reshape_2d(self.x)
        knn_classifier.fit(self.x, self.y)
        # Save the trained KNN model in the directory we provided.
        if model_dir is not None:
            with open(model_dir, 'wb') as f:
                pickle.dump(knn_classifier, f)
            print("Train completed")
            print(f"{self.missed_pictures = }")
        return knn_classifier

    @staticmethod
    def reshape_2d(data) -> list:
        """ Reshapes passed data to 2d array.
        """
        data = np.array(data)
        data_2d = data.reshape(data.shape[0], -1)
        return data_2d


if __name__ == '__main__':
    Trainer().train("training-assets", "./model/trained_knn_model.clf")  # add path here
