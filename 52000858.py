import cv2
import numpy as np
import math
import os

# Hàm so khớp biển báo với các templates
def match_with_templates(cropped_image, templates):
    max_score = 0
    matched_name = None

    # Chuyển ảnh cắt sang grayscale
    cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Multiple scales for pyramid scaling
    scales = [1.0, 0.9, 0.8, 0.7]

    for scale in scales:
        resized_cropped = cv2.resize(cropped_gray, (0, 0), fx=scale, fy=scale)

        for template_name, template_image in templates.items():
            resized_template = cv2.resize(template_image, (resized_cropped.shape[1], resized_cropped.shape[0]))

            # Template matching
            result = cv2.matchTemplate(resized_cropped, resized_template, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)

            # Update if score is highest
            if score > max_score and score > 0.6:  # Adjustable threshold
                max_score = score
                matched_name = template_name

    return matched_name

# Tăng cường độ sắc nét (sharpen)
def sharpen_image(image):
    kernel = np.array([[1, 2, 1],
                        [2, 4,2],
                        [1, 2, 1]])/16
    return cv2.filter2D(image, -1, kernel)

# Hàm để phát hiện vật thể và kiểm tra màu sắc đỏ và xanh
def detect_objects_and_validate_colors(frame, templates):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Định nghĩa khoảng màu đỏ
    lower_red = np.array([160, 100, 20])
    upper_red = np.array([180, 255, 255])
    lower_red2 = np.array([130, 30, 0])
    upper_red2 = np.array([179, 170, 255])

    # Định nghĩa khoảng màu xanh dương
    lower_blue = np.array([100, 150, 50])   # Giá trị thấp hơn H, S, V
    upper_blue = np.array([140, 255, 255]) 
    
    # Định nghĩa khoảng màu vàng (nền vàng)
    lower_yellow = np.array([0, 100, 50])
    upper_yellow = np.array([50, 255, 255])
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)

    # Tạo mask cho màu đỏ và màu vàng
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Kết hợp cả hai mặt nạ
    mask_red = cv2.inRange(hsv, lower_red, upper_red) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask_red + mask_blue + mask_yellow
    
    # Chỉ giữ lại nửa trên khung hình
    height, width = frame.shape[:2]
    mask[height // 2:, :] = 0  # Đặt mặt nạ của nửa dưới bằng 0
    # Chỉ giữ lại khoảng từ 1/3 đến 2/3 của khung hình
    mask[:, :width // 3] = 0  # Đặt mặt nạ bên trái bằng 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter))

        if 0.5 < circularity < 1:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            cropped = frame[y:y+h, x:x+w]  # Cắt vùng biển báo

            # So khớp với templates
            matched_name = match_with_templates(cropped, templates)
            if matched_name:
                cv2.putText(frame, matched_name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Kiểm tra nếu contour có 3 cạnh (tam giác)
        if len(approx) == 3:
            # Kiểm tra tỷ lệ giữa các cạnh để xác định tam giác đều
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                # Cắt vùng biển báo
                cropped = frame[y:y+h, x:x+w]

                # So khớp với templates
                matched_name = match_with_templates(cropped, templates)
                if matched_name:
                    cv2.putText(frame, matched_name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Vẽ hình chữ nhật và nhãn trên biển báo
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    
# Load tất cả các templates từ thư mục
def load_templates(template_folder):
    templates = {}
    for file_name in os.listdir(template_folder):
        if file_name.endswith(('.png', '.jpg')):
            template_path = os.path.join(template_folder, file_name)
            template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            templates[os.path.splitext(file_name)[0]] = template_image
    return templates

# Main
template_folder = 'templates'
templates = load_templates(template_folder)

cap = cv2.VideoCapture('video1.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('52000858_52000843.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    sharpened_frame = sharpen_image(frame)
    hsv = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2HSV)
    detect_objects_and_validate_colors(frame, templates)
    text = "52000858_52000843"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(frame)
    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# Đọc video thứ hai
cap = cv2.VideoCapture('video2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    sharpened_frame = sharpen_image(frame)
    hsv = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2HSV)
    detect_objects_and_validate_colors(frame, templates)
    text = "52000858_52000843"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(frame)
    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()