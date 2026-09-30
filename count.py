from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
video_path = "input/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

# Video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
out = cv2.VideoWriter(
    "output/traffic_count.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)

# Counting line
line_y = height // 2

# Already counted vehicle IDs
counted_ids = set()

# Counts
car_count = 0
motorcycle_count = 0
bus_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5],
        conf=0.3,
        verbose=False
    )

    result = results[0]

    # Draw counting line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        3
    )

    # Check tracking IDs
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            # Vehicle center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            names = {
                2: "Car",
                3: "Motorcycle",
                5: "Bus"
            }

            vehicle_name = names.get(class_id, "Vehicle")

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Vehicle ID
            cv2.putText(
                frame,
                f"{vehicle_name} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Count vehicle after crossing line
            if center_y > line_y and track_id not in counted_ids:

                counted_ids.add(track_id)

                if class_id == 2:
                    car_count += 1

                elif class_id == 3:
                    motorcycle_count += 1

                elif class_id == 5:
                    bus_count += 1

    # Display counts
    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycle_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {bus_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Save video
    out.write(frame)


# Release video
cap.release()
out.release()

print()
print("================================")
print("DONE | Vehicle counting complete")
print("Cars:", car_count)
print("Motorcycles:", motorcycle_count)
print("Buses:", bus_count)
print("Output: output/traffic_count.avi")
print("================================")