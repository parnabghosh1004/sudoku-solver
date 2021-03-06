import cv2
import numpy as np
from SudokuExtractor import extract_sudoku
import tensorflow as tf
from sudoku_algo import solve_board
from PIL import Image, ImageDraw, ImageFont
import time
from tkinter import Tk, Label, Button, Canvas, Frame
from tkinter import filedialog

model = tf.keras.models.load_model('test_model.h5')

def predict_digit(image):
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)
    image = image/255
    pred = model.predict(image)
    pred = pred[0][1:]
    if (pred[pred.argmax()]) > 0.8:
        return pred.argmax()
    return -1

def extract_digits(sudoku,digits):
    sudoku = cv2.resize(sudoku, (450, 450))
    grid = np.zeros([9, 9])
    for i in range(9):
        for j in range(9):
            if digits[i*9 + j].sum() > 10:
                image = sudoku[i*50:(i+1)*50, j*50:(j+1)*50]
                predict = predict_digit(image)
                if predict == -1:
                    return None
                grid[i][j] = predict + 1                       
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
                      str(soln), fill=(font_color if len(sub.shape) > 2 else 0), font=fnt)
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            img_solved.append(cv2_img)
    return img_solved


def inverse_perspective(img, dst_img, pts):
    pts_source = np.array([[0, 0], [img.shape[1] - 1, 0], [img.shape[1] - 1, img.shape[0] - 1], [0, img.shape[0] - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img

def simulate_solve(img_arr,warped_img,original,corners):
    for i,img in enumerate(img_arr):
        warped_img[(i//9)*img.shape[0] : (i//9+1)*img.shape[0],(i%9)*img.shape[0] : (i%9+1)*img.shape[0],:] = img
        warped_inverse = inverse_perspective(warped_img, original, np.array(corners))
        cv2.imshow('solved',warped_inverse)
        cv2.waitKey(50)
    return warped_inverse

def solve_img(img, color,camera=False):
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    extracted_img, digits,corners, warped_img, original = extract_sudoku(img.copy(), img_gray)
    board = extract_digits(extracted_img,digits)
    print(board)
    if board is None or not np.count_nonzero(board):
        return None
    unsolved_board = board.copy()
    if(not solve_board(board)):
        return None

    subd = subdivide(warped_img)
    warped_img = cv2.resize(warped_img,(9*subd[0].shape[0],9*subd[0].shape[0]))
    img_solved = put_soln(subd, unsolved_board, board,color, "arial.ttf")
    
    if not camera:
        cv2.imshow("unsolved",img)

    return simulate_solve(img_solved,warped_img,original,corners)


def captureFromWebcam():
    cam = cv2.VideoCapture(0)
    a = True
    while a:
        _,frame = cam.read()
        cv2.imshow('sudoku',frame)
        final_img = solve_img(frame,'red',True)
        if final_img is None:
            print("cannot be solved")
        else:
            cv2.destroyAllWindows()
            cv2.imshow("sudoku", final_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        k = cv2.waitKey(1)
        if k%256 == 27:
            break
    cam.release()
    cv2.destroyAllWindows()    

def UploadFromDevice():
    path = filedialog.askopenfilename()
    if path is None:
        print("select a valid file")
        return
    size = 540
    img = cv2.resize(cv2.imread(path), (size, size))
    final_img = solve_img(img.copy(), "red")
    if type(final_img).__module__ == np.__name__:    
        cv2.imshow("solved", final_img)
        cv2.waitKey(0)
    else:
        print("cannot be solved")

def OpenCamera():
    captureFromWebcam()

root = Tk()

canvas = Canvas(root,height=700,width=700,bg="#263D42")
canvas.pack()
frame = Frame(root, bg="white")
frame.place(relwidth=0.8,relheight=0.8,relx=0.1,rely=0.1)
label = Label(frame, text="SUDOKU SOLVER", fg="black",height=20)
label.pack()
button1 = Button(frame, text="Select File From Device", command=UploadFromDevice,padx=20,pady=10,fg="white",bg="#263D42")
button2 = Button(frame, text="Upload From Camera", command=OpenCamera,padx=20,pady=10,fg="white",bg="#263D42")
button1.pack()
button2.pack()

root.mainloop()
cv2.destroyAllWindows()

