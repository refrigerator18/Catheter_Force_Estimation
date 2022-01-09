import cv2
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})



# Reading sample catheter, a_10
img = cv2.imread("a_10.JPG", cv2.IMREAD_GRAYSCALE)
ret,img  = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
print(img.shape)
blank_image = np.copy(img) * 0 
diameter = 1

l, r, mid = [], [], []

# Looping through image from top left corner
for i in range(len(img)):
	for j in range(len(img[0])):
		if img[i][j] == 255:
			l.append([i, j])
			blank_image[i][j] = 255
			break

# Looping from image from bottom right corner
for i in range(len(img) - 1, -1, -1):
	for j in range(len(img[i]) -1, -1, -1):
		if img[i][j] == 255:
			# Skips top right side of catheter
			if j > 300 and j <= r[-1][1] :
					continue
			r.append([i, j])
			blank_image[i][j] = 255
			break

# Calculating middle point
test = l[::-1]
for val1, val2 in zip(test, r):
	x1, y1 = val1
	x2, y2 = val2
	x_avg = (x1+x2) / 2
	y_avg = (y1+y2) /2
	mid.append([x_avg, y_avg])
	blank_image[int(x_avg)][int(y_avg)] = 255


x_1, y_1 = l[-1]
x_2, y_2 = r[0]



# Calculating real word distance
mm_per_pixel = diameter / (y_2 - y_1) 

l = [li[::-1] for li in l]
r = [li[::-1] for li in r]
mid = [li[::-1] for li in mid]

l = np.array(l)
r = np.array(r)
mid = np.array(mid)

l = l * mm_per_pixel
r = r * mm_per_pixel
mid = mid * mm_per_pixel


# Plotting our data
# Left is blue, right is orange
plt.scatter(*zip(*l), s=0.1)
plt.scatter(*zip(*r), s=0.1)
plt.scatter(*zip(*mid), s=0.1)

plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")

plt.title("Real world coordinates for outer, middle and inner lines of catheter sample.")
plt.axis([-5, 30, -5, 30])
plt.gca().invert_yaxis()
plt.savefig('paper.png', dpi=600)
plt.show()
  

cv2.imwrite("test.JPG", blank_image)