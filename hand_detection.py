# -*- coding: utf-8 -*-
"""
Hand detection and landmark drawing using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import math


class HandDetector:
    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        debug=False,
    ):
        """
        Initialize MediaPipe hands detection
        """
        self.mp_hands = mp.solutions.hands # MediaPipe hands detection
        self.mp_drawing = mp.solutions.drawing_utils # MediaPipe drawing utils
        self.mp_drawing_styles = mp.solutions.drawing_styles # MediaPipe drawing styles
        self.debug = debug # Debug mode

        # Initialize MediaPipe hands detection object
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_and_draw_hands(self, np_img):
        """
        Detect hands in the image and draw landmarks and connections
        Also detect the salute gesture with feedback

        Args:
            np_img: Input image as numpy array (RGB format)

        Returns:
            Tuple: (processed_image, salute_detected)
        """

        # Convert RGB to BGR for MediaPipe processing
        img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Process the image with MediaPipe hands detection
        results = self.hands.process(img_bgr)

        # Convert back to RGB for drawing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        vulcan_salute_detected = False
        letter_m_detected = False

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract landmark coordinates for gesture detection
                h, w, _ = np_img.shape
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z
                    landmarks.append((x, y, z))

                # Check for Vulcan salute
                if self.is_vulcan_salute(landmarks):
                    vulcan_salute_detected = True

                    # Get handedness
                    handedness = results.multi_handedness[idx].classification[0].label

                    # Draw a special indicator for Vulcan salute
                    # Draw a rectangle around the hand
                    x_coords = [lm[0] for lm in landmarks]
                    y_coords = [lm[1] for lm in landmarks]
                    x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
                    y_min, y_max = min(y_coords) - 20, max(y_coords) + 20

                    # Draw green rectangle for Vulcan salute
                    cv2.rectangle(
                        img_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3
                    )

                    # Add text label
                    cv2.putText(
                        img_rgb,
                        f"VULCAN SALUTE ({handedness})",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                # Check for letter M gesture
                if self.is_letter_m_down(landmarks):
                    letter_m_detected = True

                    # Get handedness
                    handedness = results.multi_handedness[idx].classification[0].label

                    # Draw a special indicator for letter M
                    x_coords = [lm[0] for lm in landmarks]
                    y_coords = [lm[1] for lm in landmarks]
                    x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
                    y_min, y_max = min(y_coords) - 20, max(y_coords) + 20

                    # Draw blue rectangle for letter M
                    cv2.rectangle(
                        img_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3
                    )

                    # Add text label
                    cv2.putText(
                        img_rgb,
                        f"LETTER M ({handedness})",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2,
                    )

        return img_rgb, vulcan_salute_detected, letter_m_detected

    def get_hand_landmarks(self, np_img):
        # Convert RGB to BGR for MediaPipe processing
        img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Process the image
        results = self.hands.process(img_bgr)

        hands_data = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (left/right)
                handedness = results.multi_handedness[idx].classification[0].label

                # Extract landmark coordinates
                landmarks = []
                h, w, _ = np_img.shape
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z
                    landmarks.append((x, y, z))

                hands_data.append({"handedness": handedness, "landmarks": landmarks})

        return hands_data

    def distance(self, p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 

    # function to calculate angle between three points
    def angle_between_points(self, p1, p2, p3):
        # Vector from p2 to p1 and p2 to p3
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Calculate dot product and magnitudes
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        mag2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0

        # Calculate angle in degrees
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle = math.acos(cos_angle) * 180 / math.pi
        return angle
    
    def get_finger_average(self, tip, mcp):
        return ((tip[0] + mcp[0]) / 2, (tip[1] + mcp[1]) / 2)
    
    def is_vulcan_salute(self, landmarks):
        """
        Detect if the hand is making a Vulcan salute gesture

        The Vulcan salute is characterized by:
        - Index and middle fingers extended and together
        - Ring and pinky fingers extended and together
        - Clear separation between middle and ring fingers
        - Thumb extended/separated

        Args:
            landmarks: List of landmark coordinates [(x, y, z), ...]

        Returns:
            Boolean indicating if Vulcan salute is detected
        """
        if len(landmarks) < 21:  # MediaPipe hands has 21 landmarks
            return False

        # MediaPipe landmark indices
        WRIST = 0
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20

        INDEX_MCP = 5
        MIDDLE_MCP = 9
        RING_MCP = 13
        PINKY_MCP = 17

        # Get landmark coordinates
        wrist = landmarks[WRIST]
        index_tip = landmarks[INDEX_TIP]
        middle_tip = landmarks[MIDDLE_TIP]
        ring_tip = landmarks[RING_TIP]
        pinky_tip = landmarks[PINKY_TIP]

        index_mcp = landmarks[INDEX_MCP]
        middle_mcp = landmarks[MIDDLE_MCP]
        ring_mcp = landmarks[RING_MCP]
        pinky_mcp = landmarks[PINKY_MCP]


        

        # Check 1: Are the main fingers extended?
        # Fingers are extended if tips are further from wrist than MCPs
        wrist_to_index_tip = self.distance(wrist, index_tip)
        wrist_to_index_mcp = self.distance(wrist, index_mcp)

        wrist_to_middle_tip = self.distance(wrist, middle_tip)
        wrist_to_middle_mcp = self.distance(wrist, middle_mcp)

        wrist_to_ring_tip = self.distance(wrist, ring_tip)
        wrist_to_ring_mcp = self.distance(wrist, ring_mcp)

        wrist_to_pinky_tip = self.distance(wrist, pinky_tip)
        wrist_to_pinky_mcp = self.distance(wrist, pinky_mcp)

        fingers_extended = (
            wrist_to_index_tip > wrist_to_index_mcp * 1.1
            and wrist_to_middle_tip > wrist_to_middle_mcp * 1.1
            and wrist_to_ring_tip > wrist_to_ring_mcp * 1.1
            and wrist_to_pinky_tip > wrist_to_pinky_mcp * 1.1
        )

        if not fingers_extended:
            return False

        # Check 2: Index and middle fingers should be close together
        index_middle_distance = self.distance(index_tip, middle_tip)

        # Check 3: Ring and pinky fingers should be close together
        ring_pinky_distance = self.distance(ring_tip, pinky_tip)

        # Check 4: There should be a significant gap between middle and ring fingers
        middle_ring_distance = self.distance(middle_tip, ring_tip)

        # Check 5: The separation should create a V-shape
        # Calculate the angle between the finger groups
        middle_point_index_middle = (
            (index_tip[0] + middle_tip[0]) // 2,
            (index_tip[1] + middle_tip[1]) // 2,
        )
        middle_point_ring_pinky = (
            (ring_tip[0] + pinky_tip[0]) // 2,
            (ring_tip[1] + pinky_tip[1]) // 2,
        )

        # The angle between the two finger groups should be significant
        separation_angle = self.angle_between_points(
            middle_point_index_middle, wrist, middle_point_ring_pinky
        )

        # Define thresholds (tuned for accuracy at various distances)
        max_finger_group_distance = 70  # pixels (tightened for better accuracy)
        min_separation_distance = 50  # pixels
        min_separation_angle = 20  # degrees
        max_separation_angle = 80  # degrees

        # Additional ratio-based checks for better accuracy
        # The separation should be significantly larger than finger group distances
        min_separation_ratio = (
            1.2  # separation should be at least 1.2x larger than finger groups
        )

        # Vulcan salute conditions
        close_finger_groups = (
            index_middle_distance < max_finger_group_distance
            and ring_pinky_distance < max_finger_group_distance
        )

        good_separation = (
            middle_ring_distance > min_separation_distance
            and min_separation_angle < separation_angle < max_separation_angle
        )

        # Additional check: separation should be significantly larger than finger group distances
        max_group_distance = max(index_middle_distance, ring_pinky_distance)
        separation_ratio_check = middle_ring_distance > (
            max_group_distance * min_separation_ratio
        )

        #Debug output
        if self.debug:
            print("================ VULCAN SALUTE DEBUG ================")
            print(f"  Fingers extended: {fingers_extended}")
            print(
                f"  Index-Middle distance: {index_middle_distance:.1f} (max: {max_finger_group_distance})"
            )
            print(
                f"  Ring-Pinky distance: {ring_pinky_distance:.1f} (max: {max_finger_group_distance})"
            )
            print(
                f"  Middle-Ring separation: {middle_ring_distance:.1f} (min: {min_separation_distance})"
            )
            print(
                f"  Separation angle: {separation_angle:.1f}° (range: {min_separation_angle}-{max_separation_angle})"
            )
            print(f"  Max group distance: {max_group_distance:.1f}")
            print(
                f"  Separation ratio: {middle_ring_distance/max_group_distance:.2f} (min: {min_separation_ratio})"
            )
            print(f"  Close finger groups: {close_finger_groups}")
            print(f"  Good separation: {good_separation}")
            print(f"  Separation ratio check: {separation_ratio_check}")
            print(f"  Result: {close_finger_groups and good_separation and separation_ratio_check}")
            print("=====================================================")
            print("---")

        return close_finger_groups and good_separation and separation_ratio_check

    def is_letter_m_down(self, landmarks):
        """
        Detect if the hand is making the letter M gesture with fingers pointing down

        The letter M gesture is characterized by:
        - Index, middle and ring fingers extended and pointing down
        - Thumb and pinky NOT extended and pointing down
        - Middle finger average point should be lower than ring finger, which should be lower than index finger

        Args:
            landmarks: List of landmark coordinates [(x, y, z), ...]

        Returns:
            Boolean indicating if letter M gesture is detected
        """
        if len(landmarks) < 21:  # MediaPipe hands has 21 landmarks
            return False

        # MediaPipe landmark indices
        WRIST = 0
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16

        INDEX_MCP = 5
        MIDDLE_MCP = 9
        RING_MCP = 13

        # Get landmark coordinates
        wrist = landmarks[WRIST]
        index_tip = landmarks[INDEX_TIP]
        middle_tip = landmarks[MIDDLE_TIP]
        ring_tip = landmarks[RING_TIP]

        index_mcp = landmarks[INDEX_MCP]
        middle_mcp = landmarks[MIDDLE_MCP]
        ring_mcp = landmarks[RING_MCP]

        # Calculate average points for each finger
        index_avg = self.get_finger_average(index_tip, index_mcp)
        middle_avg = self.get_finger_average(middle_tip, middle_mcp)
        ring_avg = self.get_finger_average(ring_tip, ring_mcp)

        # Check if fingers are pointing down
        fingers_down = (
            index_tip[1] > index_mcp[1] and
            middle_tip[1] > middle_mcp[1] and
            ring_tip[1] > ring_mcp[1]
        )

        # Check if fingers are extended
        fingers_extended = (
            self.distance(wrist, index_tip) > self.distance(wrist, index_mcp) * 1.1 and
            self.distance(wrist, middle_tip) > self.distance(wrist, middle_mcp) * 1.1 and
            self.distance(wrist, ring_tip) > self.distance(wrist, ring_mcp) * 1.1
        )


        # Check the relative positions of the finger averages
        # For a downward M, middle should be lowest, then ring, then index
        correct_finger_order = (
            middle_avg[1] > ring_avg[1] >= index_avg[1]
        )

        result = fingers_down  and fingers_extended and correct_finger_order

        # Debug output
        if self.debug:
            print("================ LETTER M DOWN DEBUG ================")
            print(f"  Fingers down: {fingers_down}")
            print(f"  Fingers extended: {fingers_extended}")
            print(f"  Correct finger order: {correct_finger_order}")
            print(f"  Index average y: {index_avg[1]:.1f}")
            print(f"  Middle average y: {middle_avg[1]:.1f}")
            print(f"  Ring average y: {ring_avg[1]:.1f}")
            print(f"  Result: {result}")
            print("=====================================================")
            print("---")

        return result

    def close(self):
        """Clean up resources"""
        self.hands.close()
