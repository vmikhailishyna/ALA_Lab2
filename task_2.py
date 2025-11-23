import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#Task 1
image_raw = imread('photo.jpg')
plt.title('Original Image')
plt.imshow(image_raw)
plt.show()

vector = list(image_raw.shape)
print("Vector that contain: image dimensions in pixels and the number of main color channels used: ", vector)

#Task 2
image_sum = image_raw.sum(axis=2)
vector_converted = list(image_sum.shape)
print("Vector of converted image: ", vector_converted)
image_bw = image_sum/image_sum.max()
print(image_bw.max())
plt.title('Converted Image')
plt.imshow(image_bw, cmap='gray')
plt.show()

#Task 3.1.
pca = PCA()
pca.fit(image_bw)
variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)
cover_95 = np.argmax(cumulative_variance >= 0.95) + 1
print("Number of components needed to cover 95% of the variance:", cover_95)

#Task 3.2.
pca_1 = PCA(n_components=cover_95)
X_pca_1 = pca_1.fit_transform(image_bw)
X_reconstructed = pca_1.inverse_transform(X_pca_1)
plt.title('Resulting image for 95% data coverage,')
plt.imshow(X_reconstructed, cmap='gray')
plt.show()


#Task 4
pca_2 = PCA(n_components=5)
X_pca_2 = pca_2.fit_transform(image_bw)
X_reconstructed_2 = pca_2.inverse_transform(X_pca_2)
plt.title('Resulting image for 5 components data coverage,')
plt.imshow(X_reconstructed_2, cmap='gray')
plt.show()

pca_3 = PCA(n_components=10)
X_pca_3 = pca_3.fit_transform(image_bw)
X_reconstructed_3 = pca_3.inverse_transform(X_pca_3)
plt.title('Resulting image for 10 components data coverage,')
plt.imshow(X_reconstructed_3, cmap='gray')
plt.show()

pca_4 = PCA(n_components=30)
X_pca_4 = pca_4.fit_transform(image_bw)
X_reconstructed_4 = pca_4.inverse_transform(X_pca_4)
plt.title('Resulting image for 30 components data coverage,')
plt.imshow(X_reconstructed_4, cmap='gray')
plt.show()

pca_5 = PCA(n_components=50)
X_pca_5 = pca_5.fit_transform(image_bw)
X_reconstructed_5 = pca_5.inverse_transform(X_pca_5)
plt.title('Resulting image for 50 components data coverage,')
plt.imshow(X_reconstructed_5, cmap='gray')
plt.show()

pca_6 = PCA(n_components=150)
X_pca_6 = pca_6.fit_transform(image_bw)
X_reconstructed_6 = pca_6.inverse_transform(X_pca_6)
plt.title('Resulting image for 150 components data coverage,')
plt.imshow(X_reconstructed_6, cmap='gray')
plt.show()

pca_7 = PCA(n_components=500)
X_pca_7 = pca_7.fit_transform(image_bw)
X_reconstructed_7 = pca_7.inverse_transform(X_pca_7)
plt.title('Resulting image for 500 components data coverage,')
plt.imshow(X_reconstructed_7, cmap='gray')
plt.show()

pca_8 = PCA(n_components=740)
X_pca_8 = pca_8.fit_transform(image_bw)
X_reconstructed_8 = pca_8.inverse_transform(X_pca_8)
plt.title('Resulting image for 740 components data coverage,')
plt.imshow(X_reconstructed_8, cmap='gray')
plt.show()
