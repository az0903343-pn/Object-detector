# =========================================================================
# PRODUCTION STREAMLIT CODE: UI + REAL-TIME 3D CAMERA MESH + DECISION ENGINE
# =========================================================================
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
from ultralytics import YOLO
import numpy as np

# Streamlit Page Design Optimization Layout
st.set_page_config(page_title="Ultra 3D Classifier", layout="wide")
st.title("🤖 Ultra Pro 3D Mesh Engine & Decision Tree Classifier")
st.write("Live webcam object tracking and recognition with simulated 3D vector space graphics.")

@st.cache_resource
def load_vision_weights():
    """
    Safely loads the weight file downloaded from Kaggle.
    If 'best.pt' exists, it will use your custom weights. Otherwise, it falls back to 'yolov8n.pt'.
    """
    try:
        return YOLO("best.pt")
    except:
        return YOLO("yolov8n.pt")

vision_model = load_vision_weights()

def run_decision_tree_classifier(frame_inventory):
    """
    Algorithmic execution of a multi-node Decision Tree Split.
    Classifies the current scanned environment based on item density.
    """
    # Categorizing detected items for node split evaluation
    computer_hardware = frame_inventory.get('laptop', 0) + frame_inventory.get('keyboard', 0) + frame_inventory.get('mouse', 0)
    classroom_furniture = frame_inventory.get('chair', 0) + frame_inventory.get('book', 0) + frame_inventory.get('dining table', 0)
    home_appliances = frame_inventory.get('refrigerator', 0) + frame_inventory.get('microwave', 0) + frame_inventory.get('oven', 0) + frame_inventory.get('tv', 0)

    # Decision Tree Node 1 Split: IT Hardware density
    if computer_hardware >= 2:
        return "🌲 DECISION VERDICT: COMPUTER LABORATORY WORKSPACE"
    
    # Decision Tree Node 2 Split: Domestic appliance cluster
    elif home_appliances >= 1 and computer_hardware == 0:
        return "🌲 DECISION VERDICT: SMART HOME APPLIANCE ZONE"
    
    # Decision Tree Node 3 Split: Educational infrastructure density
    elif classroom_furniture >= 1:
        return "🌲 DECISION VERDICT: CLASSROOM / ACADEMIC ENVIRONMENT"
        
    else:
        return "🔍 ENGINE STATUS: SCANNING ENVIRONMENT..."

class Isometric3DGraphicsProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert streamlit stream frame to numpy array format
        video_frame = frame.to_ndarray(format="bgr24")
        
        # Run inference using the loaded model weights
        prediction_results = vision_model(video_frame, conf=0.40, stream=True)
        
        current_frame_inventory = {}
        
        for frame_data in prediction_results:
            bounding_boxes = frame_data.boxes
            for individual_box in bounding_boxes:
                class_index = int(individual_box.cls)
                detected_name = vision_model.names[class_index]
                
                # Keep tracking item count for the Decision Tree model pipeline
                current_frame_inventory[detected_name] = current_frame_inventory.get(detected_name, 0) + 1
                
                # Extract primary 2D corner pixels
                x_start, y_start, x_end, y_end = map(int, individual_box.xyxy)
                box_confidence = float(individual_box.conf)
                
                # ====================================================
                # ULTRA PRO 3D MODEL ISOMETRIC VECTOR GRAPHICS ENGINE
                # ====================================================
                box_width = x_end - x_start
                box_height = y_end - y_start
                
                # Mathematical projection factor to build the back depth plane of the 3D box
                depth_offset = int(min(box_width, box_height) * 0.25)
                
                # Define 3D corner mapping matrix systems (Front plane & Back plane)
                front_face = [(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)]
                back_face = [
                    (x_start + depth_offset, y_start - depth_offset),
                    (x_end + depth_offset, y_start - depth_offset),
                    (x_end + depth_offset, y_end - depth_offset),
                    (x_start + depth_offset, y_end - depth_offset)
                ]
                
                # 1. Render Back Plane Projection Face Geometry (Deep Blue Tint)
                cv2.rectangle(video_frame, back_face[0], back_face[2], (180, 105, 50), 1)
                
                # 2. Draw Connecting Isometric Depth Vector Lines (3D Mesh Wireframe Effect)
                for corner in range(4):
                    cv2.line(video_frame, front_face[corner], back_face[corner], (255, 255, 0), 2)
                
                # 3. Render Front Facing Primary Screen Geometry (Bright Neon Cyan Glow)
                cv2.rectangle(video_frame, front_face[0], front_face[2], (255, 255, 0), 2)
                
                # Cyberpunk Interactive HUD Overlay Text UI for class detection label
                overlay_string = f"[{detected_name.upper()}] Conf: {box_confidence * 100:.1f}%"
                cv2.putText(video_frame, overlay_string, (x_start, y_start - 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        # Trigger background pipeline processing through our Decision Tree Model
        ml_system_verdict = run_decision_tree_classifier(current_frame_inventory)
        
        # Draw Real-time HUD Status Data Dashboard Board Matrix Panel at the top of the screen
        cv2.rectangle(video_frame, (15, 15), (650, 60), (0, 0, 0), -1) # Dark HUD background block
        cv2.rectangle(video_frame, (15, 15), (650, 60), (255, 255, 0), 1)  # Border alignment element
        cv2.putText(video_frame, ml_system_verdict, (25, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        
        return frame.from_ndarray(video_frame, format="bgr24")

# Launch direct Client Camera Stream framework portal configuration
webrtc_streamer(
    key="isometric-3d-decision-module",
    video_processor_factory=Isometric3DGraphicsProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
