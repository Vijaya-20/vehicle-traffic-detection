from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="input/traffic.mp4",
    save=True,
    conf=0.25,
    show_labels=True,
    show_conf=True,
    line_width=3,
    project="output",
    name="traffic_boxes",
    exist_ok=True
)

print("DONE! Open the output folder.")