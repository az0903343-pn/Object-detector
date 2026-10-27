# =========================================================================
# STANDARD SYSTEM PRODUCTION GATEWAY: GRAPHICS RUNTIME COMPATIBILITY FIX
# =========================================================================
import os
import sys
import subprocess

# Secure framework installer step to initialize modules directly on runtime cluster
def secure_subsystem_boot():
    required_packages = {
        "cv2": "opencv-python-headless",
        "streamlit_webrtc": "streamlit-webrtc",
        "ultralytics": "ultralytics"
    }
    for module_name, package_target in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_target, "--no-cache-dir"])

# Run initial execution mapping setup safely
secure_subsystem_boot()

import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Setup dashboard architecture layouts
st.set_page_config(page_title="Ultra 3D Engine", layout="wide")
st.title("🤖 Ultra Pro 3D Mesh Engine & Decision Tree Classifier")
st.write("Live webcam object tracking and recognition with simulated 3D vector space graphics.")

@st.cache_resource
def load_vision_weights():
    try:
        return YOLO("best.pt")
    except:
        return YOLO("yolov8n.pt")

vision_model = load_vision_weights()

def run_decision_tree_classifier(frame_inventory):
    computer_hardware = frame_inventory.get('laptop', 0) + frame_inventory.get('keyboard', 0) + frame_inventory.get('mouse', 0)
    classroom_furniture = frame_inventory.get('chair', 0) + frame_inventory.get('book', 0) + frame_inventory.get('dining table', 0)
    home_appliances = frame_inventory.get('refrigerator', 0) + frame_inventory.get('microwave', 0) + frame_inventory.get('oven', 0) + frame_inventory.get('tv', 0)

    if computer_hardware >= 2:
        return "🌲 DECISION VERDICT: COMPUTER LABORATORY WORKSPACE"
    elif home_appliances >= 1 and computer_hardware == 0:
        return "🌲 DECISION VERDICT: SMART HOME APPLIANCE ZONE"
    elif classroom_furniture >= 1:
        return "🌲 DECISION VERDICT: CLASSROOM / ACADEMIC ENVIRONMENT"
    else:
        return "🔍 ENGINE STATUS: ACTIVE BACKGROUND SCANNING MODE..."

class Isometric3DGraphicsProcessor(VideoProcessorBase):
    def recv(self, frame):
        video_frame = frame.to_ndarray(format="bgr24")
        prediction_results = vision_model(video_frame, conf=0.40, stream=True)
        current_frame_inventory = {}
        
        for frame_data in prediction_results:
            bounding_boxes = frame_data.boxes
            for individual_box in bounding_boxes:
                class_index = int(individual_box.cls)
                detected_name = vision_model.names[class_index]
                current_frame_inventory[detected_name] = current_frame_inventory.get(detected_name, 0) + 1
                
                x_start, y_start, x_end, y_end = map(int, individual_box.xyxy)
                box_confidence = float(individual_box.conf)
                
                # --- ULTRA PRO 3D MODEL ISOMETRIC VECTOR GRAPHICS ENGINE ---
                box_width = x_end - x_start
                box_height = y_end - y_start
                depth_offset = int(min(box_width, box_height) * 0.25)
                
                pt1_f, pt2_f = (x_start, y_start), (x_end, y_end)
                pt1_b = (x_start + depth_offset, y_start - depth_offset)
                pt2_b = (x_end + depth_offset, y_end - depth_offset)
                
                cv2.rectangle(video_frame, pt1_b, pt2_b, (180, 105, 50), 1)
                cv2.line(video_frame, pt1_f, pt1_b, (255, 255, 0), 2)
                cv2.line(video_frame, (x_end, y_start), (x_end + depth_offset, y_start - depth_offset), (255, 255, 0), 2)
                cv2.line(video_frame, (x_start, y_end), (x_start + depth_offset, y_end - depth_offset), (255, 255, 0), 2)
                cv2.line(video_frame, pt2_f, pt2_b, (255, 255, 0), 2)
                cv2.rectangle(video_frame, pt1_f, pt2_f, (255, 255, 0), 2)
                
                overlay_string = f"[{detected_name.upper()}] Conf: {box_confidence * 100:.1f}%"
                cv2.putText(video_frame, overlay_string, (x_start, y_start - 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        ml_system_verdict = run_decision_tree_classifier(current_frame_inventory)
        cv2.rectangle(video_frame, (15, 15), (650, 60), (0, 0, 0), -1)
        cv2.rectangle(video_frame, (15, 15), (650, 60), (255, 255, 0), 1)
        cv2.putText(video_frame, ml_system_verdict, (25, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        
        return frame.from_ndarray(video_frame, format="bgr24")

webrtc_streamer(
    key="isometric-3d-decision-module",
    video_processor_factory=Isometric3DGraphicsProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
