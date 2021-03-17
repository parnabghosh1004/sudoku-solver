# Sudoku Solver Robotics Project
 
 ##  SUDOKU EXTRACTION -
 <table>
  <tr>
    <td>
 input image
    </td>
    <td>
 <img src="https://user-images.githubusercontent.com/58443282/111513091-72645200-8776-11eb-919f-ae110f724264.png" width="300">
    </td>
    </tr>
  </table>
  
## Preprocessing of the image-

Converts grey scaled version Original image into using a gaussian blur 
function, adaptive thresholding, and dilation to expose an image’s main features.
```def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)

    return proc
    
```
    
<table>
  <tr>
    <td>
 Preprocessed image
    </td>
    <td>
 <img src="https://user-images.githubusercontent.com/58443282/111514145-75ac0d80-8777-11eb-866d-e0dbbecdac72.png" width="300">
    </td>
    </tr>
  </table>
  
  ## FInding the contour of the largest polygon-
  
  We are finding the four extreme corners of the most prominent contour in the image.
  The image which we pre-processed is used here
  ```
  def find_corners_of_largest_polygon(img):
    opencv_version = cv2.__version__.split('.')[0]
    if opencv_version == '3':
            p
        _, contours, h = cv2.findContours(
            img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, h = cv2.findContours(
            img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    contours = sorted(contours, key=cv2.contourArea,
                      reverse=True)
    polygon = contours[0]
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1]
                                 for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1]
                                    for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]),key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
  
 
 ```
 
 ## Cropping and warping of the image-
 I am cropping and warping a rectangular section from an image into a square of similar size.
 ```
 def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[
        0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right,
                    bottom_left], dtype='float32')
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                    [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))
    
 ```
  ## Infer grid from the square image
  
  Infers 81 cell grid from a square image
  
  ```
  def infer_grid(img):
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    return squares
  ```
  ## Get each digit
  The next step is to extract digits from their cells and build an array.
  
  ```
  def get_digits(img, squares, size):
    digits = []
    img = pre_process_image(img.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits
  ```
  
 <table>
  <tr>
    <td>
 solved image
    </td>
    <td>
 <img src="https://user-images.githubusercontent.com/58443282/111514875-2adec580-8778-11eb-973c-2593963a4d6b.png" width="300">
    </td>
    </tr>
  </table> 
  
  ## RENDERING THE SOLUTION -
  we divide the warped image into small sub-images, then using PIL, we draw the digit in the square using a solved array. 
  Finally, combine all the squares into the picture (warped image), 
  using inverse homographic transform to transform the warped image’s plane into the original image.
  

```
def inverse_perspective(img, dst_img, pts):
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], 
                          [0, img.shape[0] - 1]],dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img
 ```
   <table>
  <tr>
    <td>
 final image
    </td>
    <td>
 <img src="https://user-images.githubusercontent.com/58443282/111515344-ae98b200-8778-11eb-87c7-99d70269df7a.png" width="300">
    </td>
    </tr>
  </table> 
  
  


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  


  
