if __name__ == '__main__':
    from models.yolo_detector import YOLODetector
    import cv2

    from models.yolo_detector import YOLODetector
    
    from pathlib import Path
    detector = YOLODetector(str(Path(__file__).resolve().parent / "best.pt"))

    img = cv2.imread("frame_phone.jpg")

    # достаточно низкий порог, чтобы увидеть хоть что-то
    results = detector.detect(img, conf_threshold=0.2)
    print(results)  # посмотрим, что вообще приходит

    violations = detector.get_violations(results)
    print("violations:", len(violations))
    for v in violations:
        print(v["class_id"], v["class_name"], v["confidence"])
