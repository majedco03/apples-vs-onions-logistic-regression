
class Main:
    # interface for main application logic with a menu for options
    
    @staticmethod
    def run():
        from Model import Model
        from Data import Data
        import numpy as np
        from PIL import Image
        import os

        # Initialize the model
        model = Model()
        trained = False
        X_train, Y_train, X_test, Y_test = None, None, None, None

        while True:
            print("\n=== Apples vs Onions Classifier Menu ===")
            print("1. Train the model")
            print("2. Calculate model accuracy")
            print("3. Classify an image")
            print("4. Exit")
            choice = input("Enter your choice (1-4): ").strip()

            if choice == '1':
                print("Training the model...")
                X_train, Y_train, X_test, Y_test = model.train()
                trained = True
                print("Model trained successfully!")

            elif choice == '2':
                if not trained:
                    print("Please train the model first (option 1).")
                    continue
                print("Calculating accuracies...")
                train_accuracy = model.evaluate(X_train, Y_train)
                test_accuracy = model.evaluate(X_test, Y_test)
                print(f"Training Accuracy: {train_accuracy:.2f}%")
                print(f"Test Accuracy: {test_accuracy:.2f}%")

            elif choice == '3':
                if not trained:
                    print("Please train the model first (option 1).")
                    continue
                image_path = input("Enter the path of the image to classify (apple/onion): ").strip().strip("'\"")
                if not os.path.isfile(image_path):
                    print("The provided path does not exist or is not a file.")
                    continue
                # Load and preprocess the image
                img = Image.open(image_path).convert('RGB').resize((64, 64))  # Convert to RGB and resize to 64x64
                img_array = np.array(img)
                img_flat = img_array.flatten().reshape(-1, 1)  # Reshape to (features, 1)
                # Make prediction
                prediction = model.predict(img_flat)
                # Output result
                if prediction[0, 0] == 1:
                    print("The image is classified as: Apple")
                elif prediction[0, 0] == 0:
                    print("The image is classified as: Onion")

            elif choice == '4':
                print("Exiting the application. Goodbye!")
                break

            else:
                print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    Main.run()