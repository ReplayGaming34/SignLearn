import cv2
import numpy as np
import tensorflow as tf

from processing import extract_normalized_landmarks  # Reuse your logic!


class App:
    def start(self):
        # 1. Load the model and your label map
        model = tf.keras.models.load_model("models/sign_language_alphabet_model.h5")
        # Ensure this list matches the folder order from processing
        labels = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
        ]

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Process frame for landmarks
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            coords = extract_normalized_landmarks(image_rgb)

            if coords:
                # 3. Reshape for the model (Samples, Features, Channels)
                input_data = np.array(coords).reshape(1, 42, 1)

                # 4. Predict
                prediction = model.predict(input_data, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                # 5. Display if confidence is high enough
                if confidence > 0.8:
                    print(class_id)
                    letter = labels[class_id]
                    cv2.putText(
                        frame,
                        f"{letter} ({confidence:.2f})",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Sign Language Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
