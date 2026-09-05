import cv2
import numpy as np

# ==== Change this path ====
image_path = "/home/ashu/my_robot_ws/src/simple_bumperbot_py_examples/simple_bumperbot_py_examples/Screenshot from 2026-04-23 15-15-51.png"
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not read image from", image_path)
    exit()

# Initial HSV values
l_h, l_s, l_v = 0, 0, 0
u_h, u_s, u_v = 179, 255, 255

print("Controls:")
print("Q/A: LH + / -")
print("W/S: LS + / -")
print("E/D: LV + / -")
print("R/F: UH + / -")
print("T/G: US + / -")
print("Y/H: UV + / -")
print("ESC: Exit")

while True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display HSV values on result image
    display = result.copy()
    text = f"L:({l_h},{l_s},{l_v}) U:({u_h},{u_s},{u_v})"
    cv2.putText(display, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Result", display)

    key = cv2.waitKey(50) & 0xFF

    # Lower HSV adjustments
    if key == ord('q'): l_h = min(l_h + 1, 179)
    if key == ord('a'): l_h = max(l_h - 1, 0)

    if key == ord('w'): l_s = min(l_s + 1, 255)
    if key == ord('s'): l_s = max(l_s - 1, 0)

    if key == ord('e'): l_v = min(l_v + 1, 255)
    if key == ord('d'): l_v = max(l_v - 1, 0)

    # Upper HSV adjustments
    if key == ord('r'): u_h = min(u_h + 1, 179)
    if key == ord('f'): u_h = max(u_h - 1, 0)

    if key == ord('t'): u_s = min(u_s + 1, 255)
    if key == ord('g'): u_s = max(u_s - 1, 0)

    if key == ord('y'): u_v = min(u_v + 1, 255)
    if key == ord('h'): u_v = max(u_v - 1, 0)

    if key == 27:
        break

cv2.destroyAllWindows()