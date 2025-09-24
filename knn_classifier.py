import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

# Loading the data
# ---------------------------------------------------------------------------------------
data_dir = "data"
categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

X = []    # features, or image data
y = []    # labels, or categories

for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for filename in os.listdir(folder_path):    
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).resize((64,64))  # resize
            img_array = np.array(img) / 255.0  # normalize to [0, 1]
            img_array = img_array.flatten()  # flatten to 1D array
            X.append(img_array)  
            y.append(label) 

X = np.array(X)
y = np.array(y)


# Can add PCA for dimensionality reduction here later
# ---------------------------------------------------------------------------------------
pca_full = PCA()
pca_full.fit(X)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# Plot cumulative variance to pick components
plt.figure(figsize=(8,5))
plt.plot(cum_var, marker='o')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance')
plt.title('Choosing number of PCA components')
plt.grid(True)
plt.show()

# Choose components that explain ~90-95% variance (adjust based on plot)
n_components = 150
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)



# Split the data
# ---------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the KNN classifier
# ---------------------------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)  # Can fine tune the number of neighbors
knn.fit(X_train, y_train)


# Evaluate the model
# ---------------------------------------------------------------------------------------
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories))

# Predicting new images
# ---------------------------------------------------------------------------------------
def predict_image(image_path):
    img = Image.open(img_path).convert("RGB") 
    img = img.resize((64, 64), Image.Resampling.LANCZOS)    
    img_array = np.array(img) / 255.0
    img_array = img_array.flatten().reshape(1, -1)  # reshape for prediction
    img_pca = pca.transform(img_array)  # project to PCA space
    prediction = knn.predict(img_array)
    return categories[prediction[0]]



if __name__ == "__main__":
    # Example: single image prediction
    image_path = "test_imgs/bike.jpeg"
    predicted_class = predict_image(image_path)
    print(f"The predicted class is: {predicted_class}")

    # Example: batch prediction
    import os
    folder_path = "test_imgs"
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            predicted_class = predict_image(img_path)
            print(f"{filename}: {predicted_class}")