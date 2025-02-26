from ultralytics import YOLO
model=YOLO('yolov8s.pt')

results=model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='train_dataset'
)