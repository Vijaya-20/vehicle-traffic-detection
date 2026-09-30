from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Track vehicles using ByteTrack
results = model.track(
    source="input/traffic.mp4",
    tracker="bytetrack.yaml",
    classes=[2, 3, 5, 7],   # car, motorcycle, bus, truck
    conf=0.3,
    save=True,
    project="runs",
    name="tracking",
    show=False
)

print("DONE | Vehicle tracking completed.")
print("Open: runs/tracking")