from project_functions import *

test_image = mpimg.imread('./vehicles/GTI_Far/image0000.png')
feature_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
features, vis = get_hog_features(feature_gray, 9, 8, 2, vis=True)

fig = plt.figure()
plt.subplot(121)
plt.imshow(test_image, cmap='gray')
plt.title('Vehicle image, grayscaled')
plt.subplot(122)
plt.imshow(vis, cmap='gray')
plt.title('HOG of vehicle')
fig.tight_layout()
plt.show()