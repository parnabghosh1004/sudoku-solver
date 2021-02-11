final_img = extract_sudoku('sudoku.jpg')
final_img = cv2.resize(final_img, (450, 450))
board = extract_digits(final_img)

solved_board = solve_board(board)

print(solved_board)

cv2.imshow("img", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
