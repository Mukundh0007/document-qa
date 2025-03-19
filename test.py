
#-----------------------| IMPORTS |--------------------------
import time
import queue
import threading
import cv2
import requests
# from vidgear.gears import NetGear
from flask import Flask, Response, request, render_template, url_for, jsonify
from ultralytics import YOLO
import ultralytics
import numpy as np
import os
from flask_socketio import SocketIO
import subprocess

#---------------------| INITIALIZATIONS |---------------------
ultralytics.checks()

model = YOLO('C:\projects\MJ_MSIL\MSIL_new_sealant.pt', task='segment')
claw_model = YOLO( r'C:\projects\MJ_MSIL\MSIL_new_claw.pt', task='detect')


logo = cv2.imread("defect-scanner-logo-transparent-cropped.png")
BATCH_FILE_PATH = r"C:\projects\new.bat"   #give the batch file path
capFrame = logo.copy()
st_f = 0
claw_results = claw_model.predict(
    logo, conf=0.2, iou=0.4, imgsz=640, verbose=False)
model.predict(
    logo, conf=0.2, iou=0.4, imgsz=640, verbose=False)
click_position = 0
frame_counter = 0
model_prefix = "YEDD"
old_car_model = ""
interlock_f = 0

model_dict = {
    "YEDH1C2EP74": "YEDD",
    "YEDH1C2GP74": "YEDD",
    "YEDH1C21P74": "YEDD",
    "YEDJ1C9EP68": "YEDE",
    "YEDJ1C9EP80": "YEDE",
    "YEDJ1C9GP06": "YEDE",
    "YEDJ1C9GP07": "YEDE",
    "YEDJ1C9GP47": "YEDE",
    "YEDJ1C9GP68": "YEDE",
    "YEDJ1C9GP80": "YEDE",
    "YEDJ1C9JP07": "YEDE",
    "YEDJ1C9JP47": "YEDE",
    "YEDJ1C9JP68": "YEDE",
    "YEDJ1C9JP80": "YEDE",
    "YEDK1C2CP06": "YEDE",
    "YEDK1C2CP74": "YEDD",
    "YEDK1C2EP68": "YEDE",
    "YEDK1C2EP74": "YEDD",
    "YEDK1C2EP80": "YEDE",
    "YEDK1C2GP06": "YEDE",
    "YEDK1C2GP68": "YEDE",
    "YEDK1C2GP74": "YEDD",
    "YEDK1C2GP80": "YEDE",
    "YEDK1C2JP68": "YEDE",
    "YEDK1C2JP80": "YEDE",
    "YEDK1C21P74": "YEDD",
    "YEDK1C4EP74": "YEDD",
    "YEDK1C4GP74": "YEDD",
    "YEDK112JP74": "YEDD",
    "YEDK114JP74": "YEDD",
    "YEDK2C2CP74": "YEDD",
    "YEDK2C2EP74": "YEDD",
    "YEDK2C2GP74": "YEDD",
    "YEDK2C21P74": "YEDD",
    "YEDK2C4EP74": "YEDD",
    "YEDK2C4GP74": "YEDD",
    "YEDK212JP74": "YEDD",
    "YEDK214JP74": "YEDD",
    "YEDL2C2EP74": "YEDD",
    "YEDL2C2GP74": "YEDD",
    "YEDL2C21P74": "YEDD",
    "YEDM1C2EP77": "YEDE",
    "YEDM1C2EP96": "YEDE",
    "YEDM1C2JP77": "YEDE",
    "YEDM1C2JP96": "YEDE",
    "YEDM1C9EP77": "YEDE",
    "YEDM1C9EP85": "YEDE",
    "YEDM1C9EP96": "YEDE",
    "YEDM1C9JP07": "YEDE",
    "YEDM1C9JP77": "YEDE",
    "YEDM1C9JP96": "YEDE",
    "YEDM1C2EP9612708": "YEDE",
    "YEDM1C2JP9613991": "YEDE",
    "YEDM1C9EP9612707": "YEDE",
    "YEDM1C9EP9612708": "YEDE",
    "YEDM1C9JP9613990": "YEDE",
    "YEDM1C9JP9613991": "YEDE",
    "YEDM1C9EP8512707": "YEDE",
    "YEDM1C2JP9613990": "YEDE"
}



imgid = ['000216', '000450', '000550', '000750', '001']
# images_dir = []
diff = {0: "Processing", 1: "Processing", 2: "Processing", 3: "Processing"}
diffpercent = {0: "Processing", 1: "Processing",
               2: "Processing", 3: "Processing"}
d = {0: [], 1: [], 2: []}

# UI FONT specifications
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2  # Increased font size by 1 unit
color = (0, 255, 0)  # Green color in BGR, defined
thickness = 2


Counter = 0
status = "Processing"
Ground_Truth = cv2.imread(r"C:\projects\gt_images\frame_001_binary.png", 0)
Predicted = Ground_Truth
missing_mask = Ground_Truth
bottom_right = Ground_Truth
id = imgid[Counter]
binary_mask = np.zeros((384, 640), dtype=np.uint8)

Car_Model = ""

loc = threading.Lock()  # Semaphore()
dummy_img = cv2.imread("defect_scanner_logo.png")
logo = cv2.imread("defect-scanner-logo-transparent-cropped.png")

frame_ori = dummy_img.copy()
options = {"max_retries": 2, "request_timeout": 20, }
app = Flask(__name__, )
socketio = SocketIO(app)

def create_composite_frame(frame, box1, box2, box3, box4):
    """
    Creates a 1920×1080 composite frame:
    - Resizes the main frame proportionally into a 960×1080 area with black padding.
    - Stacks 4 boxes of 960×270 on the right.
    """

    # Original dimensions of the frame (always 1920×1080)
    h, w = frame.shape[:2]
    aspect_ratio = w / h

    # Calculate new size while maintaining aspect ratio (resize width to 960, keep aspect ratio)
    new_width = 960
    new_height = int(960 / aspect_ratio)

    # If new height is greater than 1080, resize based on height instead
    if new_height > 1080:
        new_height = 1080
        new_width = int(1080 * aspect_ratio)

    # Resize the main frame while maintaining aspect ratio
    main_frame_resized = cv2.resize(frame, (new_width, new_height))

    # Create a black image (960×1080) for padding
    main_frame_padded = np.zeros((1080, 960, 3), dtype=np.uint8)

    # Calculate padding offsets to center the resized image in the black background
    x_offset = (960 - new_width) // 2
    y_offset = (1080 - new_height) // 2

    # Place resized frame in the center of the padded black image
    main_frame_padded[y_offset:y_offset + new_height,
    x_offset:x_offset + new_width] = main_frame_resized

    # Create a blank 1920×1080 image for the final layout
    composite = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Place the main frame (960×1080) on the left side
    composite[0:1080, 0:960] = main_frame_padded

    # Resize the four boxes to 960×270 each
    box1_resized = cv2.resize(box1, (960, 270))
    box2_resized = cv2.resize(box2, (960, 270))
    box3_resized = cv2.resize(box3, (960, 270))
    box4_resized = cv2.resize(box4, (960, 270))

    # Convert grayscale to BGR if needed
    if len(box1_resized.shape) == 2:
        box1_resized = cv2.cvtColor(box1_resized, cv2.COLOR_GRAY2BGR)
    if len(box2_resized.shape) == 2:
        box2_resized = cv2.cvtColor(box2_resized, cv2.COLOR_GRAY2BGR)
    if len(box3_resized.shape) == 2:
        box3_resized = cv2.cvtColor(box3_resized, cv2.COLOR_GRAY2BGR)
    if len(box4_resized.shape) == 2:
        box4_resized = cv2.cvtColor(box4_resized, cv2.COLOR_GRAY2BGR)

    # Place four boxes on the right side (960×270 each)
    composite[0:270, 960:1920] = box1_resized
    composite[270:540, 960:1920] = box2_resized
    composite[540:810, 960:1920] = box3_resized
    composite[810:1080, 960:1920] = box4_resized

    return composite

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


@app.route("/favicon.ico")
def favicon():
    return url_for('static', filename='data:,')


@app.route('/')
def index():
    print("here")
    return "Maruti Suzuki"


def shift_mask(mask, shift_y):
    """
    Moves the mask up or down by shift_y pixels.

    :param mask: Input binary mask (numpy array).
    :param shift_y: Number of pixels to move (+ve for down, -ve for up).
    :return: Shifted mask.
    """
    try:
        print("Shift y ->", shift_y)
        h, w = mask.shape[:2]  # Get mask dimensions

        # Define translation matrix: Move by (0, shift_y)
        M = np.float32([[1, 0, 0], [0, 1, shift_y]])

        # Apply translation
        # Black padding for shifted area
        shifted_mask = cv2.warpAffine(mask, M, (w, h), borderValue=0)

        return shifted_mask
    except Exception as e:
        print("Error in shift_mask ->", e)


def predict_and_matchGT(frame, gt_name, shift_position):
    global binary_mask, model
    try:
        frame = cv2.resize(frame, (1920, 1080))
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results_seg = model.predict(
            source=frame, conf=0.05, iou=0.8, imgsz=640, verbose=False, device=[0])
        # frame = results_seg[0].plot(conf=True, labels=True, boxes=False)
        Predicted = results_seg[0].plot(conf=True, labels=True, boxes=False)
        predicted_classes = results_seg[0].boxes.cls.cpu().numpy()
        # Initialize an accumulated mask
        accumulated_mask = np.zeros_like(binary_mask, dtype=np.uint8)

        for j, a in enumerate(predicted_classes):
            mask = results_seg[0].masks[j].data.cpu().numpy()[0]
            binary_mask = (mask * 255).astype(np.uint8)
            # Add the current binary mask to the accumulated mask
            accumulated_mask = cv2.add(accumulated_mask, binary_mask)
        Ground_Truth = cv2.imread(f"./ground_truth/{gt_name}.png", 0)
        Ground_Truth = shift_mask(Ground_Truth,shift_position)
        # Identify missing pixels (where ground has object but resized_accumulated_mask doesn't)
        missing_mask = cv2.bitwise_and(Ground_Truth, cv2.bitwise_not(
            cv2.resize(accumulated_mask, (960, 540))))


        actualW = np.count_nonzero(Ground_Truth == 255)
        differenceW = np.count_nonzero(missing_mask == 255)
        percentage = round((differenceW / actualW) * 100, 2)
        print(percentage)
        if percentage < 35:
            percent_flag = True
        else:
            percent_flag = False
        return Ground_Truth, Predicted, missing_mask, percent_flag
    except Exception as e:
        print("There is an error in predict_and_matchGT: ", e)
        return None, None, None, None


def frameInf(frame):
    global model, claw_model
    global dummy_img, click_position
    global logo
    global imgid
    global diff
    global diffpercent
    global d

    global font
    global fontScale
    global color
    global thickness
    global Counter
    global Ground_Truth
    global Predicted
    global missing_mask
    global bottom_right
    global binary_mask
    global Car_Model, old_car_model, model_dict
    global frame_counter, model_prefix

    # frame = cv2.resize(frame, (640, 640))
    if frame is None:
        return (dummy_img)
    frame_counter += 1
    ori_frame_ = frame.copy()
    # print("2", frame.shape)

    # cv2.imwrite(fr"C:\projects\18_2_25\{time.time()}_{Car_Model}.png",ori_frame_)

    claw_results = claw_model.predict(
        frame, conf=0.1, iou=0.4, imgsz=640, verbose=False, device=[0])
    frame = claw_results[0].plot(labels=False)  # conf=False,

    all_cls = claw_results[0].boxes.cls.tolist()
    all_xy = claw_results[0].boxes.xyxy.tolist()

    if (click_position == 0 or click_position == 1) and 0 in all_cls:
        for i, val in enumerate(all_cls):
            # print("first click",all_xy[i][1])

            if val == 0:
                if click_position == 0 and 350 <= all_xy[i][1] < 370:
                    click_position += 1

                    gt_name = f"{model_prefix}_0"
                    print("1->", all_xy[i][1],
                          frame_counter, gt_name, old_car_model)
                    Ground_Truth, Predicted, missing_mask, percent_flag = predict_and_matchGT(
                        frame, gt_name, 356 - int(all_xy[i][1]))
                    if percent_flag:
                        diffpercent[0] = "OK"
                        socketio.emit('interlock', {'value': 1})
                    elif "YED" in old_car_model:
                        diffpercent[0] = "NOT OK"
                        socketio.emit('interlock', {'value': 2})

                    break

                elif click_position == 1 and 135 <= all_xy[i][1] < 155:
                    click_position += 1
                    gt_name = f"{model_prefix}_1"
                    print("2->", all_xy[i][1],
                          frame_counter, gt_name, old_car_model)
                    Ground_Truth, Predicted, missing_mask, percent_flag = predict_and_matchGT(
                        frame, gt_name, 152 - int(all_xy[i][1]))

                    if percent_flag:
                        diffpercent[1] = "OK"
                        socketio.emit('interlock', {'value': 1})
                    elif "YED" in old_car_model:
                        diffpercent[1] = "NOT OK"
                        socketio.emit('interlock', {'value': 2})
                    break
    elif (click_position == 2 or click_position == 3 or click_position == 4) and (1 in all_cls or 2 in all_cls):
        for i, val in enumerate(all_cls):
            if val == 1 or val == 2:
                if click_position == 2 and 420 <= all_xy[i][1] < 440:
                    click_position += 1
                    gt_name = f"{model_prefix}_2"
                    print("3->", all_xy[i][1],
                          frame_counter, gt_name, old_car_model)
                    Ground_Truth, Predicted, missing_mask, percent_flag = predict_and_matchGT(
                        frame, gt_name, 438 - int(all_xy[i][1]))
                    if percent_flag:
                        diffpercent[2] = "OK"
                        socketio.emit('interlock', {'value': 1})
                    elif "YED" in old_car_model:
                        diffpercent[2] = "NOT OK"
                        socketio.emit('interlock', {'value': 2})
                    break
                elif click_position == 3 and 200 <= all_xy[i][1] < 220:
                    click_position += 1
                    gt_name = f"{model_prefix}_3"
                    print("4->", all_xy[i][1],
                          frame_counter, gt_name, old_car_model)
                    Ground_Truth, Predicted, missing_mask, percent_flag = predict_and_matchGT(
                        frame, gt_name, 219 - int(all_xy[i][1]))
                    if percent_flag:
                        diffpercent[3] = "OK"
                        socketio.emit('interlock', {'value': 1})
                    elif "YED" in old_car_model:
                        diffpercent[3] = "NOT OK"
                        socketio.emit('interlock', {'value': 2})
                    break
    if click_position != 0 and 0 in all_cls:
        for i, val in enumerate(all_cls):
            # print("first click",all_xy[i][1])
            if val == 0:
                if 400 <= all_xy[i][1] < 540:
                    print("Force reset all for next")
                    click_position = 0
                    diffpercent = {0: "Processing", 1: "Processing",
                                   2: "Processing", 3: "Processing"}
                    old_car_model = Car_Model
                    model_prefix = model_dict.get(old_car_model, "YEDD")

#Final UI areas
    img = create_composite_frame(
        frame, Ground_Truth, Predicted, missing_mask, bottom_right)
    cv2.putText(img, "Video Feed", (285, 200), font,
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "Master Image", (1300, 80), font,
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "Actual Image", (1300, 295), font,
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "Difference", (1300, 590), font,
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, f"Car Model = {Car_Model}", (285, 880),
                font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "Difference Values", (1300, 880), font,
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, f"Difference in Front: {diffpercent[0]}", (
        1100, 920), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, f"Difference in Front & Middle: {diffpercent[1]}", (
        1100, 960), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, f"Difference in Middle: {diffpercent[2]}", (
        1100, 1000), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, f"Difference in Rear: {diffpercent[3]}", (
        1100, 1040), font, fontScale, color, thickness, cv2.LINE_AA)
    img = cv2.resize(img, (960, 540))
    return img


@app.route('/sc')
def screen_shot():
    global frame_ori
    global frame_queue
    t = ThreadWithResult(
        target=frameInf, args=(frame_ori, time.time()))
    t.start()
    frame_queue.put(t)
    return "Clicked"


flag = 1
client = 0
th_l = 0
_, buffer = cv2.imencode('.jpg', dummy_img)
frame_queue = queue.Queue(90)

@app.route('/st')
def getF():
    global capFrame
    print("video start")
    video_path = "C:\projects\yed_dome_19_2_25.mp4"

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)  # cv2.CAP_FFMPEG
    while True:
        # for _ in range(5):  # Skip old frames
        #     cap.grab()
        success, frame = cap.read()
        if success:
            capFrame = frame.copy()
            cv2.waitKey(500)
        else:
            print("No frame")
            time.sleep(5)

            cap = cv2.VideoCapture(video_path)
    cap.release()
    print("video Stopped")
    return "Video Stopped"


def generate_frames():
    global buffer
    global frame_queue
    global th_l
    global st_f, capFrame
    if st_f == 0:
        try:
            requests.get("http://localhost:5000/st", timeout=2)
        except:
            pass
        st_f = 1
    while True:
        if th_l != 0:
            # print("th_l=1",end=" ")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        loc.acquire()
        th_l = 1
        # video_path = "rtsp://sourab:tvsm123!@0.tcp.in.ngrok.io:10577/stream1"
        # cap = cv2.VideoCapture(video_path)

        try:
            while True:
                res = frameInf(capFrame)
                # print(time.time())
                try:
                    _, buffer = cv2.imencode('.jpg', res)
                except Exception as e:
                    print(e)
                    pass
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            # cap.release()
            # print("Exception occur", e)
            pass
        th_l = 0
        loc.release()

@app.route('/il')
def il():
    global interlock_f
    if interlock_f == 1:
        interlock_f = 2
        print("Interlock NG send")
        socketio.emit('interlock', {'value': interlock_f})
    else:
        interlock_f = 1
        print("Interlock OK send")
        socketio.emit('interlock', {'value': interlock_f})

    return "Done"


@app.route('/live')
def index_t():
    # A simple HTML page to display the video stream.
    return """
    <html>
      <head>
        <title>Defect Scanner</title>
      </head>
      <body>
        <img src="/video_feed" style="width:80%;">
      </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/model=<int:model_id>', methods=['GET'])
def get_model_images(model_id):
    global model_prefix
    # Validate the model number
    model_prefix_types = {
        1: "YEDD",  # Model 1 images: YED_0, YED_1, YED_2, YED_3
        2: "YEDE",  # Model 1 images: YED_0, YED_1, YED_2, YED_3
        3: "YXA",  # Model 2 images: YXA_0, YXA_1, YXA_2, YXA_3
        4: "YSD",  # Model 3 images: YSD_0, YSD_1, YSD_2, YSD_3
    }
    if model_id not in model_prefix:
        return jsonify({"error": "Invalid model number. Valid options are 1, 2, or 3."}), 400

    model_prefix = model_prefix_types[model_id]

    return jsonify({
        "model": model_id,
    }), 200

    update_result = f"Update performed for model {model_id}"

    # Return a JSON response with a success message
    return jsonify({"message": update_result}), 200


@socketio.on("plc_data")
def plc_list_to_string(values):
    global Car_Model, model_prefix, model_dict, click_position, old_car_model
    if all(v == 0 for v in values):
        return ""
    result = ""
    for value in values:
        # Convert to 4-character hex string (uppercase)
        hex_value = format(value, '04X')
        # Swap byte order (little-endian)
        swapped_hex = hex_value[2:] + hex_value[:2]
        # Convert hex to ASCII
        result += bytes.fromhex(swapped_hex).decode('ASCII')

    if len(Car_Model) == 0:
        model_prefix = model_dict.get(Car_Model, "YEDD")
        old_car_model = result
    Car_Model = result


@app.route('/update')
def run_batch():
    if not os.path.exists(BATCH_FILE_PATH):
        return jsonify({"error": "Batch file not found."}), 404

    try:
        result = subprocess.run(BATCH_FILE_PATH, shell=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("EXIT CODE:", result.returncode)

        return jsonify({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        })
    except Exception as e:
        print("Exception:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False) 
