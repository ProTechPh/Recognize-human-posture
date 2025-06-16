import cv2
import time
import math as m
import mediapipe as mp
import numpy as np

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def print_num(num):
    print(num)

def sendWarning(x):
    pass

# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX
font_small = cv2.FONT_HERSHEY_SIMPLEX

# Colors - using brighter, more visible colors
blue = (255, 127, 0)
red = (0, 0, 255)  # Pure red
green = (0, 255, 0)  # Pure green
dark_blue = (127, 20, 0)
light_green = (0, 255, 127)
yellow = (0, 255, 255)
pink = (255, 0, 255)
white = (255, 255, 255)
black = (0, 0, 0)
cyan = (255, 255, 0)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class VideoCamera(object):
    def __init__(self):
        # Try to use the webcam
        try:
            self.cap = cv2.VideoCapture(0)
        except:
            print("Could not access webcam, using default camera")
            self.cap = cv2.VideoCapture(0)
            
        self.good_frames = 0
        self.bad_frames = 0
        self.last_time = time.time()
        self.start_time = time.time()
        self.posture_status = "Analyzing..."
    
    def __del__(self):
        self.cap.release()
    
    def add_status_panel(self, image, neck_angle, torso_angle, offset, good_time, bad_time, correct_percent):
        h, w = image.shape[:2]
        
        # Create a semi-transparent overlay for the status panel
        overlay = image.copy()
        panel_height = 100  # Even smaller panel height
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        
        # Add the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Add status text with a more visible background
        is_good_posture = neck_angle < 40 and torso_angle < 10
        status_text = "GOOD POSTURE" if is_good_posture else "POOR POSTURE"
        
        # Make status text more proportional
        text_size = cv2.getTextSize(status_text, font, 0.7, 1)[0]
        text_x = (w - text_size[0]) // 2  # Center the text
        text_y = h - panel_height + 25
        
        # Draw status background - full width but not too tall
        status_bg_color = (0, 100, 0) if is_good_posture else (100, 0, 0)
        cv2.rectangle(image, 
                     (0, text_y - 20), 
                     (w, text_y + 5), 
                     status_bg_color, -1)
        
        # Draw status text - centered and properly sized
        cv2.putText(image, status_text, (text_x, text_y), 
                   font, 0.7, white, 1, cv2.LINE_AA)
        
        # Add time stats with better spacing and smaller text
        y_offset = h - panel_height + 55
        
        # Left column labels
        cv2.putText(image, "Good:", (10, y_offset), 
                   font, 0.5, green, 1, cv2.LINE_AA)
        cv2.putText(image, "Bad:", (10, y_offset + 20), 
                   font, 0.5, red, 1, cv2.LINE_AA)
        cv2.putText(image, "Total:", (10, y_offset + 40), 
                   font, 0.5, cyan, 1, cv2.LINE_AA)
        
        # Right column values - aligned
        cv2.putText(image, f"{round(good_time, 1)}s", (80, y_offset), 
                   font, 0.5, white, 1, cv2.LINE_AA)
        cv2.putText(image, f"{round(bad_time, 1)}s", (80, y_offset + 20), 
                   font, 0.5, white, 1, cv2.LINE_AA)
        cv2.putText(image, f"{round(good_time + bad_time, 1)}s", (80, y_offset + 40), 
                   font, 0.5, white, 1, cv2.LINE_AA)
        
        # Add percentage with better proportions
        percent_text = f"{round(correct_percent, 1)}%"
        percent_size = cv2.getTextSize(percent_text, font, 0.7, 1)[0]
        percent_x = w - percent_size[0] - 15
        percent_y = h - 30
        
        # Draw percentage with colored background based on value
        percent_bg_color = (0, 100, 0) if correct_percent >= 70 else \
                          (100, 100, 0) if correct_percent >= 40 else \
                          (100, 0, 0)
                          
        # Make percentage box more proportional
        cv2.rectangle(image, 
                     (w - 100, percent_y - 20), 
                     (w - 10, percent_y + 5), 
                     percent_bg_color, -1)
                     
        cv2.putText(image, percent_text, (percent_x, percent_y), 
                   font, 0.7, white, 1, cv2.LINE_AA)
        
        # Add alignment status with better proportions
        align_text = f"Align: {'Good' if offset < 100 else 'Adjust'}"
        align_color = green if offset < 100 else yellow
        
        # Position alignment status on the right side
        align_x = w - 150
        align_y = y_offset + 20
        
        cv2.putText(image, align_text, (align_x, align_y), 
                   font, 0.5, align_color, 1, cv2.LINE_AA)
        
        return image
    
    def draw_angle_visualization(self, image, neck_angle, torso_angle, landmarks):
        h, w = image.shape[:2]
        
        # Draw angle arcs for visualization
        l_shldr_x = int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
        l_shldr_y = int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        l_ear_x = int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * w)
        l_ear_y = int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * h)
        l_hip_x = int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * w)
        l_hip_y = int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * h)
        
        # Draw angle arc for neck - more proportional size
        neck_color = green if neck_angle < 40 else red
        cv2.ellipse(image, 
                   (l_shldr_x, l_shldr_y), 
                   (25, 25),  # Even smaller radius
                   0, 
                   -90, 
                   -90 + neck_angle, 
                   neck_color, 
                   1)  # Thinner line
        
        # Draw angle arc for torso
        torso_color = green if torso_angle < 10 else red
        cv2.ellipse(image, 
                   (l_hip_x, l_hip_y), 
                   (25, 25),  # Even smaller radius
                   0, 
                   -90, 
                   -90 + torso_angle, 
                   torso_color, 
                   1)  # Thinner line
        
        # Add angle labels with proportional size
        neck_text = f"{int(neck_angle)}°"
        text_size = cv2.getTextSize(neck_text, font, 0.5, 1)[0]
        text_x = l_shldr_x + 10
        text_y = l_shldr_y - 10
        
        # Draw background rectangle - proportional
        cv2.rectangle(image, 
                     (text_x - 3, text_y - text_size[1] - 3), 
                     (text_x + text_size[0] + 3, text_y + 3), 
                     (0, 0, 0), -1)
        
        # Draw text - smaller
        cv2.putText(image, neck_text, (text_x, text_y), 
                   font, 0.5, neck_color, 1, cv2.LINE_AA)
        
        # Same for torso angle
        torso_text = f"{int(torso_angle)}°"
        text_size = cv2.getTextSize(torso_text, font, 0.5, 1)[0]
        text_x = l_hip_x + 10
        text_y = l_hip_y - 10
        
        # Draw background rectangle - proportional
        cv2.rectangle(image, 
                     (text_x - 3, text_y - text_size[1] - 3), 
                     (text_x + text_size[0] + 3, text_y + 3), 
                     (0, 0, 0), -1)
        
        # Draw text - smaller
        cv2.putText(image, torso_text, (text_x, text_y), 
                   font, 0.5, torso_color, 1, cv2.LINE_AA)
        
        return image
    
    def get_frame(self):
        success, image = self.cap.read()
        if not success:
            return [b'', 0]
            
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)*(2/4))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*(2/4))
        frame_size = (width, height)
        
        image = cv2.resize(image, (width, height),fx=0,fy=0,interpolation= cv2.INTER_LINEAR)

        fps = max(self.cap.get(cv2.CAP_PROP_FPS), 1)  # Ensure fps is at least 1
        # Get height and width.
        h, w = image.shape[:2]
        
        # Add a title bar with better proportions
        title_height = 30  # Even shorter title bar
        title_bar = np.zeros((title_height, w, 3), dtype=np.uint8)
        title_bar[:] = (45, 45, 45)  # Dark gray background
        
        # Add title with proper proportions
        title_text = "Posture Analysis"
        # Shadow
        cv2.putText(title_bar, title_text, (6, 20), 
                   font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        # Text
        cv2.putText(title_bar, title_text, (5, 20), 
                   font, 0.6, white, 1, cv2.LINE_AA)
        
        # Add current date and time with better proportions
        current_time = time.strftime("%H:%M:%S", time.localtime())
        time_text = f"{current_time}"
        time_width = cv2.getTextSize(time_text, font, 0.5, 1)[0][0]
        cv2.putText(title_bar, time_text, (w - time_width - 5, 20), 
                   font, 0.5, white, 1, cv2.LINE_AA)
        
        # Combine title bar with image
        image_with_title = np.vstack((title_bar, image))
        
        # Update image reference to include title bar
        image = image_with_title
        h, w = image.shape[:2]

        # Process the image for pose detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Check if landmarks were detected
        if keypoints.pose_landmarks is None:
            # Create a semi-transparent overlay
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            
            # Display a message when no person is detected - more proportional
            message = "No person detected"
            text_size = cv2.getTextSize(message, font, 0.8, 1)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + title_height) // 2
            
            # Add background for better visibility
            cv2.rectangle(image,
                         (text_x - 10, text_y - 25),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 150), -1)
            
            cv2.putText(image, message, (text_x, text_y), 
                       font, 0.8, (255, 165, 0), 1, cv2.LINE_AA)
            
            instruction = "Please stand in front of the camera"
            inst_size = cv2.getTextSize(instruction, font, 0.6, 1)[0]
            inst_x = (w - inst_size[0]) // 2
            inst_y = text_y + 40
            
            cv2.putText(image, instruction, (inst_x, inst_y), 
                       font, 0.6, (255, 200, 0), 1, cv2.LINE_AA)
            
            # Convert to JPEG
            ret, jpeg = cv2.imencode('.jpg', image)
            return [jpeg.tobytes(), image]

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        try:
            # Draw pose landmarks with custom settings for better visibility
            custom_connections = [
                (lmPose.LEFT_SHOULDER, lmPose.LEFT_ELBOW),
                (lmPose.LEFT_ELBOW, lmPose.LEFT_WRIST),
                (lmPose.RIGHT_SHOULDER, lmPose.RIGHT_ELBOW),
                (lmPose.RIGHT_ELBOW, lmPose.RIGHT_WRIST),
                (lmPose.LEFT_SHOULDER, lmPose.RIGHT_SHOULDER),
                (lmPose.LEFT_HIP, lmPose.RIGHT_HIP),
                (lmPose.LEFT_SHOULDER, lmPose.LEFT_HIP),
                (lmPose.RIGHT_SHOULDER, lmPose.RIGHT_HIP),
                (lmPose.LEFT_HIP, lmPose.LEFT_KNEE),
                (lmPose.LEFT_KNEE, lmPose.LEFT_ANKLE),
                (lmPose.RIGHT_HIP, lmPose.RIGHT_KNEE),
                (lmPose.RIGHT_KNEE, lmPose.RIGHT_ANKLE)
            ]
            
            # Draw custom connections with proportional lines
            for connection in custom_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = (int(lm.landmark[start_idx].x * w), 
                              int(lm.landmark[start_idx].y * h))
                end_point = (int(lm.landmark[end_idx].x * w), 
                            int(lm.landmark[end_idx].y * h))
                
                cv2.line(image, start_point, end_point, (255, 255, 255), 1)
            
            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.      
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Draw landmarks with proportional sizes
            cv2.circle(image, (l_shldr_x, l_shldr_y), 5, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 5, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 5, yellow, -1)

            # Add angle visualization
            image = self.draw_angle_visualization(image, neck_inclination, torso_inclination, lm)

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if neck_inclination < 40 and torso_inclination < 10:
                self.good_frames += 1
                posture_color = green
                
                # Join landmarks with proportional lines
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 2)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 2)
                
                self.posture_status = "GOOD POSTURE"
            else:
                self.bad_frames += 1
                posture_color = red
                
                # Join landmarks with proportional lines
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 2)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 2)
                
                self.posture_status = "BAD POSTURE"

            # Calculate the time of remaining in a particular posture.
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            good_time = (1 / fps) * self.good_frames
            bad_time = (1 / fps) * self.bad_frames

            total_time = good_time + bad_time
            correct_percent = (good_time/total_time) * 100 if total_time > 0 else 0

            # Add status panel
            image = self.add_status_panel(image, neck_inclination, torso_inclination, 
                                         offset, good_time, bad_time, correct_percent)

            # If you stay in bad posture for more than 3 minutes (180s) send an alert.
            if bad_time > 180:
                sendWarning(1)
                
        except Exception as e:
            # Handle any errors in landmark detection or processing
            error_msg = f"Error: {str(e)}"
            # Add background for better visibility
            cv2.rectangle(image,
                         (10, 30 + title_height - 20),
                         (350, 70 + title_height),
                         (50, 0, 0), -1)
                         
            cv2.putText(image, error_msg, (20, 30 + title_height), 
                       font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Please ensure your body is visible", 
                       (20, 60 + title_height), font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', image)
        return [jpeg.tobytes(), image]
