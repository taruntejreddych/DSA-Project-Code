import matplotlib.pyplot as plt 
import numpy as np 

#READING THE IMAGE INTO AN ARRAY
img = plt.imread("img.jpg")
img = np.array(img)

def pcaComp(img, NumPC=64): 
### The inputs to the function are an array with the image, and the number of principal components to take while recomstructing the image.
### It returns an array with the image constructed using the given number of principal components.
	cov = img - np.mean(img , axis = 1)
	eVal, eVec = np.linalg.eigh(np.cov(cov)) #Eigenvalues and Eigenvectors of the covariance matrix

	l = eVec.shape[1]
	idx = np.argsort(eVal)
	idx = idx[::-1]
	eVec = eVec[:,idx] #Sorting the eigenalues in descending order
	eVal = eVal[idx] #Sorting the eigenvectors based on corresponding eigenvalues
	pc= NumPC
	if pc<l or pc>0:
		eVec = eVec[:, range(pc)]
	out = np.dot(eVec.T, cov)
	out = np.dot(eVec, out) + np.mean(img, axis = 1).T 
	out = np.uint8(np.absolute(out)) # Take absolute value incase of complex eigenvectors
	return out, eVal


#RECONSTRUCTING THE IMAGE WITH DIFFERENT NUMBER OF PRINCIPAL COMPONENTS
out1, eVal = pcaComp(img, 16)
out2, eVal = pcaComp(img,32)
out3, eVal = pcaComp(img,64)
plt.figure(1)
x = np.linspace(1,8,8)
(markers, stemlines, baseline) = plt.stem(x,eVal[0:8])
plt.setp(baseline, visible=False)
plt.title('Eigenvalues')
plt.ylabel('Eigenvalue')
plt.xlabel('Rank of eigenvalue in descending order')
plt.ylim(0,260000)
plt.show()
#PLOTTING IMAGES
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(2,2,2)
plt.imshow(out1, cmap='gray')
plt.title('16 Princpal Pomponents')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(2,2,3)
plt.imshow(out2, cmap='gray')
plt.title('32 Princpal Components')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(2,2,4)
plt.imshow(out3, cmap='gray')
plt.title('64 Princpal Components')
plt.show()
plt.xlabel('x')
plt.ylabel('y')