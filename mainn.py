import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize video capture
cap = cv2.VideoCapture("data/carLicence4.mp4")  

# Initialize YOLO model
model = YOLO("weights/best.pt")  

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

def clean_license_text(text):
    """Clean and format the recognized license plate text"""
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    # Common OCR corrections
    text = text.replace("O", "0").replace("I", "1").replace("Z", "2")
    return text

def recognize_plate(frame, x1, y1, x2, y2):
    """Perform OCR on license plate region"""
    plate_img = frame[y1:y2, x1:x2]
    result = ocr.ocr(plate_img, det=False, rec=True, cls=False)
    
    if result and result[0]:
        text, confidence = result[0][0]
        if confidence > 0.6:  
            return clean_license_text(text)
    return None

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect license plates
    results = model(frame, conf=0.45)
    
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Recognize license plate text
            plate_text = recognize_plate(frame, x1, y1, x2, y2)
            
            if plate_text:
                print(f"Detected License Plate: {plate_text}")
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw text background
                text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                            (x1 + text_size[0], y1), (0, 255, 0), -1)
                
                # Draw license plate text
                cv2.putText(frame, plate_text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Display the processed frame
    cv2.imshow("License Plate Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()