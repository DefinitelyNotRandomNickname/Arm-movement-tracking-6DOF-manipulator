import cv2
import mediapipe as mp
import math
import numpy as np
import serial
import time


def main():
    # Initialize MediaPipe hands model
    mp_hands_1 = mp.solutions.hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1, max_num_hands=1)
    mp_hands_2 = mp.solutions.hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1, max_num_hands=1)
    previous_elbow_1 = []
    previous_elbow_2 = []
    previous_hand_1 = []
    previous_hand_2 = []
    serial_port = 'COM7'

    # Set up video capture
    cap_1 = cv2.VideoCapture(0) # Up
    cap_2 = cv2.VideoCapture(1) # Side
    while cap_1.isOpened() and cap_2.isOpened():
        # Get images from webcams
        success_1, image_1 = cap_1.read()
        success_2, image_2 = cap_2.read()
        if not success_1 or not success_2:
            print("Failed to read video")
            return 1
        
        # Rotate the image by 180 degrees
        image_1 = cv2.rotate(image_1, cv2.ROTATE_180)
        
        # Rotate the image by 90 degrees
        image_2 = cv2.rotate(image_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Detect landmarks from both images
        image_1, landmarks_1 = detection(image_1, mp_hands_1, previous_hand_1, previous_elbow_1, "up")
        image_2, landmarks_2 = detection(image_2, mp_hands_2, previous_hand_2, previous_elbow_2, "side")
        
        if landmarks_1 and landmarks_2:
            # Merge landmarks
            landmarks = [[landmarks_1[i][0], landmarks_1[i][1], landmarks_2[i]] for i in range(len(landmarks_1))]

            # Estimate elbow position based on direction
            new_elbow(landmarks)
            
            # Draw elbow
            cv2.circle(image_1, (int(landmarks[0][0] * image_1.shape[1]), int(landmarks[0][1] * image_1.shape[0])), radius=1, color=(255, 0, 0), thickness=2)
            cv2.line(image_1, (int(landmarks[0][0] * image_1.shape[1]), int(landmarks[0][1] * image_1.shape[0])), (int(landmarks[1][0] * image_1.shape[1]), int(landmarks[1][1] * image_1.shape[0])), (0, 0, 255), 2)
            
            # Draw shoulder
            cv2.circle(image_1, (int(0.5 * image_1.shape[1]), int(1.2 * image_1.shape[0])), radius=1, color=(255, 0, 0), thickness=2)
            cv2.line(image_1, (int(0.5 * image_1.shape[1]), int(1.2 * image_1.shape[0])), (int(landmarks[0][0] * image_1.shape[1]), int(landmarks[0][1] * image_1.shape[0])), (0, 0, 255), 2)     
            
            # Calculate angles
            angles = calculate_angles(landmarks)
            
            # Calculate duties based on angles
            duties = calculate_duties(angles)
            
            # Create communication frame
            frame = createFrame(duties)
        
            try:
                # Open USB port
                ser = serial.Serial(serial_port, baudrate=115200, timeout=1)
                time.sleep(0.1)

                data = frame.strip() + "\r\n"
                
                # Send the data over the serial port.
                ser.write(data.encode())
                
                ser.flush()
                ser.close()

            except serial.SerialException as e:
                print(f"Serial port error: {e}")
            
        else:
            # No hand detected
            cv2.putText(image_1, "NO HAND DETECTED", (0, image_1.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image_2, "NO HAND DETECTED", (0, image_2.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Horizontal concatenation of images
        concatenated_image = ConcatenateImages(image_1, image_2)
        
        cv2.imshow(f"Detection", concatenated_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Clean up
    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()


"""
======================================

HAND AND LINE DETECTION

======================================
"""


"""
Select detected lines that are appropriate for the elbow direction candidate. Function handles image from upper camera.
"""
def select_lines_up(lines, wrist_y, line_endpoints):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
                    
            # Set x2 to be at the right side
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                                        
            # If line is beyond wrist ignore it
            if y2 < wrist_y or y1 < wrist_y:
                continue
            
            dx, dy = x2 - x1, y2 - y1
            
            # Extend the line
            x0, y0 = x1 + (-10000) * dx, y1 + (-10000) * dy
            
            line_endpoints.append((x0, y0))


"""
Select detected lines that are appropriate for the elbow direction candidate. Function handles image from side camera.
"""
def select_lines_side(lines, wrist_x, line_endpoints):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
                    
            # Set x2 to be at the right side
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
                                        
            # If line is beyond wrist ignore it
            if x2 > wrist_x:
                continue
            
            dx, dy = x2 - x1, y2 - y1
            
            # Extend the line
            x0, y0 = x1 + (-10000) * dx, y1 + (-10000) * dy
                        
            line_endpoints.append((x0, y0))
            

"""
Function detects hand on the image and draws it, finds elbow direction, and returns it if found. If no hand was found it 
returns previous results if there were any. If there are no previous records it returns no landmarks.
"""
def detection(image, mp_hands, previous_hand, previous_elbow, position):
    # Canny edge detection
    edges = cv2.Canny(image, 100, 100)

    # Detect the hand
    results_hands = mp_hands.process(image)
    
    # Hand detected, proceed
    if results_hands.multi_hand_landmarks:
        hand_landmarks = results_hands.multi_hand_landmarks[0]
        
        # Add found landmarks to the history
        previous_hand.append(hand_landmarks)
        
        if len(previous_hand) > 2:
            previous_hand.pop(0)

        # Iterate through the joints
        for joint_index in range(len(previous_hand[0].landmark)):
            # Get mean coords for each joint in each hand in list and write it to our landmarks
            hand_landmarks.landmark[joint_index].x = sum(element.landmark[joint_index].x for element in previous_hand) / len(previous_hand)
            hand_landmarks.landmark[joint_index].y = sum(element.landmark[joint_index].y for element in previous_hand) / len(previous_hand)
            hand_landmarks.landmark[joint_index].z = sum(element.landmark[joint_index].z for element in previous_hand) / len(previous_hand)
            
        # Replace newest element in the list with new one
        previous_hand[-1] = hand_landmarks

        # Draw hand landmarks
        mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        # Get the wrist landmark
        wrist_landmark = hand_landmarks.landmark[0]
        wrist_x, wrist_y = int(wrist_landmark.x * image.shape[1]), int(wrist_landmark.y * image.shape[0])

        # Use Hough Line Transform to detect lines
        lines =  cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 50, 10)
        lines_endpoints = []

        if position == "up":
            select_lines_up(lines, wrist_y, lines_endpoints)
        else:
            select_lines_side(lines, wrist_x, lines_endpoints)
            
        try:
            # Calculate the mean endpoint of the detected lines
            mean_x = sum(coord[0] for coord in lines_endpoints) / len(lines_endpoints)
            mean_y = sum(coord[1] for coord in lines_endpoints) / len(lines_endpoints)
            
            # Add the mean endpoint to the array
            previous_elbow.append((int(mean_x), int(mean_y)))
            if len(previous_elbow) > 10:
                previous_elbow.pop(0)
        except:
            pass
            
        landmarks = None    
        
        if len(previous_elbow) > 0:
            # Calculate the mean point of all points in array over time
            elbow_x = sum(coord[0] for coord in previous_elbow) / len(previous_elbow)
            elbow_y = sum(coord[1] for coord in previous_elbow) / len(previous_elbow)
            
            if position == "up":
                # Get X, Y coordinates of wrist, start of index finger, start of pinky, tip of thumb, tip of middle finger
                landmarks = [(hand_landmarks.landmark[index].x, hand_landmarks.landmark[index].y) for index in [0, 5, 17, 4, 12]]
                
                # Insert elbow X, Y coordinates at the start of the list
                landmarks.insert(0, (elbow_x / image.shape[1], elbow_y / image.shape[0]))
            else:
                # Get Z coordinates of wrist, start of index finger, start of pinky, tip of thumb, tip of middle finger
                landmarks = [hand_landmarks.landmark[index].y for index in [0, 5, 17, 4, 12]]
                
                # Insert elbow Z coordinate at the start of the list
                landmarks.insert(0, elbow_y / image.shape[0])
        
        return image, landmarks
    
    # Hand not detected, proceed with previous one
    elif len(previous_hand) > 0:
        hand_landmarks = previous_hand[-1]
        
        # Draw hand landmarks
        mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        # Get the wrist landmark
        wrist_landmark = hand_landmarks.landmark[0]
        wrist_x, wrist_y = int(wrist_landmark.x * image.shape[1]), int(wrist_landmark.y * image.shape[0])
        
        elbow_x, elbow_y = 0, 0
        
        if len(previous_elbow) > 0:
            # Calculate the mean point of all points in array
            elbow_x = sum(coord[0] for coord in previous_elbow) / len(previous_elbow)
            elbow_y = sum(coord[1] for coord in previous_elbow) / len(previous_elbow)
        
        if position == "up":
            # Get X, Y coordinates of wrist, start of index finger, start of pinky, tip of thumb, tip of middle finger
            landmarks = [(hand_landmarks.landmark[index].x, hand_landmarks.landmark[index].y) for index in [0, 5, 17, 4, 12]]
            
            # Insert elbow X, Y coordinates at the start of the list
            landmarks.insert(0, (elbow_x / image.shape[1], elbow_y / image.shape[0]))
        else:
            # Get Z coordinates of wrist, start of index finger, start of pinky, tip of thumb, tip of middle finger
            landmarks = [hand_landmarks.landmark[index].y for index in [0, 5, 17, 4, 12]]
            
            # Insert elbow Z coordinate at the start of the list
            landmarks.insert(0, elbow_y / image.shape[0])
        
        return image, landmarks
    
    else:
        return image, None


"""
======================================

ELBOW POSITION ESTIMATION

======================================
"""


"""
Calculate where elbow is placed based on distance between start of index finger and wrist. This function uses dependency that distance between wrist
and elbow is around twice and a half as long as distance between start of index finger and wrist. Then replace elbow landmark with a new one.
"""
def new_elbow(landmarks):
    # Calculate delta between wrist and start of index finger
    delta_length = np.array([landmarks[1][0] - landmarks[2][0], landmarks[1][1] - landmarks[2][1], landmarks[1][2] - landmarks[2][2]])
    
    # Calculate it's length
    length = np.linalg.norm(delta_length)
    
    # Get coordinates of elbow and wrist
    elbow_x, elbow_y, elbow_z = landmarks[0]
    wrist_x, wrist_y, wrist_z = landmarks[1]
        
    # Calculate the interpolation factor
    factor = length * 2.6 / math.sqrt((wrist_x - elbow_x)**2 + (wrist_y - elbow_y)**2 + (wrist_z - elbow_z)**2)

    new_x = wrist_x + factor * (elbow_x - wrist_x)
    new_y = wrist_y + factor * (elbow_y - wrist_y)
    new_z = wrist_z + factor * (elbow_z - wrist_z)
    
    # Cap new coordinates within the image
    new_x = max(0.0, min(1.0, new_x))
    new_y = max(0.0, min(1.5, new_y)) # Except elbow can be invisible in Y coord
    new_z = max(0.0, min(1.0, new_z))
    
    # Replace old coordinates with new ones
    landmarks[0] = (new_x, new_y, new_z)


"""
======================================

JOINTS' ANGLE CALCULATION

======================================
"""


"""
Calculate angle between OY and projected on XY plane upperarm.
"""
def angle_0(landmarks):
    vector_0Y = np.array([0.0, 1.0, 0.0])
    upperarm = np.array([landmarks[0][0], 1.5 - landmarks[0][1], 1.0 - landmarks[0][2]])
    
    # Factor of OZ and elbow, and their length
    factor = np.dot(vector_0Y, upperarm)
    length_0 = np.linalg.norm(vector_0Y)
    length_1 = math.sqrt((upperarm[0] - 0.5)**2 + upperarm[1]**2)
    
    # Angle between vector and OZ axis
    angle_0 = np.degrees(np.arccos(factor / (length_0 * length_1)))
    
    if upperarm[0] > 0.5:
        angle_0 += 90
    else:
        angle_0 = 90 - angle_0
    
    return angle_0


"""
Calculate angle between OZ and upperarm.
"""
def angle_1(landmarks):
    vector_0Z = np.array([0.0, 0.0, 1.0])
    upperarm = np.array([abs(landmarks[0][0] - 0.5), 1.5 - landmarks[0][1], abs(landmarks[0][2] - 0.5)])

    # Factor of OY and upperarm, and their length
    factor = np.dot(vector_0Z, upperarm)
    length_0 = np.linalg.norm(vector_0Z)
    length_1 = np.linalg.norm(upperarm)
    
    # Angle between vector and OY on plane [X, Y, Z]
    angle_1 = np.degrees(np.arccos(factor / (length_0 * length_1)))
    
    if landmarks[0][2] < 0.5:
        angle_1 = 180 - angle_1
    else:
        angle_1 = 90 - (90 - angle_1) * 1.8

    return angle_1


"""
Calculate angle between lower and upper arm.
"""
def angle_2(landmarks):
    upperarm = np.array([landmarks[0][0] - 0.5, 1.5 - landmarks[0][1], 0.5 - landmarks[0][2]])
    forearm = np.array([-landmarks[0][0] + landmarks[1][0], landmarks[0][1] - landmarks[1][1], landmarks[0][2] - landmarks[1][2]])
    
    # Factor of forearm and upperarm, and their length
    factor = np.dot(upperarm, forearm)
    length_0 = np.linalg.norm(upperarm)
    length_1 = np.linalg.norm(forearm)

    # Calculate the angle between the vectors
    angle_2 = np.degrees(np.arccos(factor / (length_0 * length_1)))
    
    return angle_2


"""
Calculate angle between hand and forearm.
"""
def angle_3(landmarks):
    # Get two sides of hand and calculate their mean point
    middle_side = np.array([(landmarks[2][0] + landmarks[3][0]) / 2.0, (landmarks[2][1] + landmarks[3][1]) / 2.0, (landmarks[2][2] + landmarks[3][2]) / 2.0])
    
    forearm = np.array([-landmarks[0][0] + landmarks[1][0], landmarks[0][1] - landmarks[1][1], landmarks[0][2] - landmarks[1][2]])
    hand = np.array([-landmarks[1][0] + middle_side[0], landmarks[1][1] - middle_side[1], landmarks[1][2] - middle_side[2]])
    
    # Factor of hand and forearm, and their length
    factor = np.dot(forearm, hand)
    length_0 = np.linalg.norm(forearm)
    length_1 = np.linalg.norm(hand)

    # Calculate the angle between the vectors
    angle_3 = np.degrees(np.arccos(factor / (length_0 * length_1)))
    
    # Chech if hand vector is at the right side of forearm vector
    if np.sum(np.cross(forearm, hand)) < 0.0:
        angle_3 += 90
    else:
        angle_3 = 90 - angle_3

    return angle_3


"""
Calculate angle between vector created from hand orientation and XY plane.
"""
def angle_4(landmarks):    
    hand = np.array([-landmarks[2][0] + landmarks[3][0], landmarks[2][1] - landmarks[3][1], landmarks[2][2] - landmarks[3][2]])
    
    # Normal vector of [X, Y] plane
    vector_0Z = np.array([0.0, 0.0, 1.0])
    
    # Factor of hand and OZ, and their length
    factor = np.dot(hand, vector_0Z)
    length_0 = np.linalg.norm(hand)
    length_1 = np.linalg.norm(vector_0Z)
    
    # Calculate the angle between the line and plane
    angle_4 = np.degrees(np.arccos(factor / (length_0 * length_1)))
        
    return angle_4


"""
Based on distance between fingers, based on hand length calculate theoretical angle.
"""
def angle_5(landmarks):
    wrist = np.array(landmarks[1])
    index_side = np.array(landmarks[2])
    
    index_finger = np.array(landmarks[4])
    little_finger = np.array(landmarks[5])
    
    # Calculate deltas between wrist and start of index finger
    delta_length_x = (wrist[0] - index_side[0])
    delta_length_y = (wrist[1] - index_side[1])
    delta_length_z = (wrist[2] - index_side[2])
    
    # Calculate deltas between fingers
    delta_distance_x = (index_finger[0] - little_finger[0])
    delta_distance_y = (index_finger[1] - little_finger[1])
    delta_distance_z = (index_finger[2] - little_finger[2])
    
    # Calculate length factor
    length_factor = abs(math.sqrt(delta_length_x**2 + delta_length_y**2 + delta_length_z**2))

    # Calculate distance between fingers
    distance = abs(math.sqrt(delta_distance_x**2 + delta_distance_y**2 + delta_distance_z**2))
    
    # Get angle from between [0, 180] degrees based on distance and length of hand
    if distance <= 0:
        angle_5 = 0.0
    elif distance >= 1.5 * length_factor:
        angle_5 = 180.0
    else:
        angle_5 = (180.0 / (1.5 * length_factor)) * distance
        
    return angle_5


"""
Calculate all the angles from gotten landmarks and get em into output array
"""
def calculate_angles(landmarks):
    angles_tmp = []
    angles = []
    angles_tmp.append(angle_0(landmarks))
    angles_tmp.append(angle_1(landmarks))
    angles_tmp.append(angle_2(landmarks))
    angles_tmp.append(angle_3(landmarks))
    angles_tmp.append(angle_4(landmarks))
    angles_tmp.append(angle_5(landmarks))
    
    # Cap angles within [0.0, 180.0]
    for i in range(len(angles_tmp)):
        if i != 2 and i != 1:
            angles.append(max(0.0, min(180.0, angles_tmp[i])))
        else:
            angles.append(max(0.0, min(90.0, angles_tmp[i])))
    
    angles[0] = 180 - angles[0]
    angles[3] = 180 - angles[3]
    angles[5] = 180 - angles[5]

    return angles


"""
======================================

ANGLE TO DUTY MAPPING

======================================
"""


"""
Map angle to duty based on caps.
"""
def angle_to_duty(angle, min_duty, max_duty, index):
    # Calculate the proportion of the angle within the range
    if index != 2:
        proportion = (angle) / (180.0)
    else:
        proportion = (angle) / (90.0)
    
    # Map the proportion to the duty range
    value = min_duty + proportion * (max_duty - min_duty)
    
    return value


"""
Map all the angles to duties. Caps based on servos.
"""
def calculate_duties(angles):
    min_duties = [2100, 2600, 2000, 2000, 1500, 5500]
    max_duties = [8580, 8500, 6000, 8000, 8000, 8200]
    
    # Map angles to duties within given range
    duties = [int(angle_to_duty(angles[i], min_duties[i], max_duties[i], i)) for i in range(len(angles))]
        
    return duties


"""
======================================

COMMUNICATION FRAME CREATION

======================================
"""


"""
Calculate CRC-16 from list
"""
def calculate_crc16(data):
    crc = 0xFFFF

    for value in data:
        # Convert each value to bytes with big-endian byte order
        bytes_value = value.to_bytes(2, byteorder='big')

        for byte in bytes_value:
            # XOR'ing, Shifting, yaddi-yaddi-yadda
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1

    fin = crc & 0xFFFF
    return format(fin, '04X')


"""
Create communication frame from array of servos' duties.
"""
def createFrame(duties):
    # Calculate CRC-16 of duties
    checksum = calculate_crc16(duties)
    
    frame = f"S {checksum} "
    
    for servo in duties:
        frame += f"{servo} "
    
    return frame + f"E"


"""
======================================

IMAGE OPERATIONS

======================================
"""

"""
Merge two images into one. They are placed next to each other and centered vertically.
"""
def ConcatenateImages(image_1, image_2):
    # Get the dimensions of the images
    height_1, width_1 = image_1.shape[:2]
    height_2, width_2 = image_2.shape[:2]

    max_height = max(height_1, height_2)

    # Calculate vertical offset for each image
    offset_1 = (max_height - height_1) // 2
    offset_2 = (max_height - height_2) // 2

    # Create two black images with the maximum dimensions
    black_image_1 = np.zeros((max_height, width_1, 3), dtype=np.uint8)
    black_image_2 = np.zeros((max_height, width_2, 3), dtype=np.uint8)

    # Copy the original images onto the black backgrounds with vertical centering
    black_image_1[offset_1:offset_1 + height_1, :width_1] = image_1
    black_image_2[offset_2:offset_2 + height_2, :width_2] = image_2

    # Return horizontal concatenation of images
    return np.concatenate((black_image_1, black_image_2), axis=1)


if __name__ == "__main__":
    main()
