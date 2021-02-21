import cv2
import numpy as np
from SudokuExtractor import extract_sudoku
import tensorflow as tf
from sudoku_algo import solve_board
from PIL import Image, ImageDraw, ImageFont
import time

model = tf.keras.models.load_model('test_model.h5')


def predict_digit(image):
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)
    image = image/255
    pred = model.predict_classes(image)
    return pred[0]


def extract_digits(sudoku):
    sudoku = cv2.resize(sudoku, (450, 450))
    grid = np.zeros([9, 9])
    for i in range(9):
        for j in range(9):
            image = sudoku[i*50:(i+1)*50, j*50:(j+1)*50]
            if image.sum() > 80000:
                grid[i][j] = predict_digit(image)
    return grid.astype(int)


def subdivide(img):
    height, _ = img.shape[:2]
    box = height // 9
    subd = []
    for i in range(9):
        for j in range(9):
            subd.append(img[i*box:(i+1)*box, j*box:(j+1)*box])
    return subd


def put_soln(subd, unsolved_arr, solved_arr, font_color, font_path):
    unsolveds = np.array(unsolved_arr).reshape(81)
    solns = np.array(solved_arr).reshape(81)
    paired = list((zip(solns, unsolveds, subd)))
    img_solved = []
    for soln, unsolved, sub in paired:
        if(soln == unsolved):
            img_solved.append(sub)
        else:
            img_h, img_w = sub.shape[:2]
            img_rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            fnt = ImageFont.truetype(font_path, int(img_h/1.4))
            font_w, font_h = draw.textsize(str(soln), font=fnt)
            draw.text(((img_w - font_w) / 2, (img_h - font_h) / 2 - img_h // 10),
                      str(soln), fill=(font_color if len(img.shape) > 2 else 0), font=fnt)
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            img_solved.append(cv2_img)
    return img_solved


def stitch_image(img_arr):

    rows = {}
    for i in range(9):
        row_img = img_arr[i*9]
        for j in range(1, 9):
            row_img = np.concatenate((row_img, img_arr[i*9+j]), axis=1)
        rows[i] = row_img
    solved_img = rows[0]
    for key, value in rows.items():
        if(key != 0):
            solved_img = np.concatenate((solved_img, value), axis=0)

    return solved_img


def inverse_perspective(img, dst_img, pts):
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img


def solve_img(img, color):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    extracted_img, corners, warped_img, original = extract_sudoku(
        img, img_gray)

    board = extract_digits(extracted_img)
    unsolved_board = board.copy()
    if(not solve_board(board)):
        return None

    subd = subdivide(warped_img)
    img_solved = put_soln(subd, unsolved_board, board,
                          color, "arial.ttf")

    stitched_image = stitch_image(img_solved)
    warped_inverse = inverse_perspective(
        stitched_image, original, np.array(corners))
    
    img = warped_img
    return warped_inverse

def captureFromWebcam():
    cam = cv2.VideoCapture(0)
    a = True
    img_name = None
    while a:
        _,frame = cam.read()
        cv2.imshow('sudoku solver',frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        extracted_img, corners, warped_img, original = extract_sudoku(
        frame, frame_gray)
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            break
        elif k%256 == 32:
            img_name = 'sudoku-webcam.png'
            cv2.imwrite(img_name, frame)
            print('image captured')
            break
    cam.release()
    cv2.destroyAllWindows()
    return img_name    

path = captureFromWebcam()

if not path:
    print("Escape hit, closing...")
else:    
    size = 540
    img = cv2.resize(cv2.imread(path), (size, size))
    final_img = solve_img(img.copy(), "red")
    if type(final_img).__module__ == np.__name__:    
        cv2.imshow("unsolved", img)
        # cv2.imshow("solved", final_img)
        cv2.waitKey(0)
    else:
        print("cannot be solved")

cv2.destroyAllWindows()

