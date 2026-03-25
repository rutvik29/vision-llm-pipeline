"""CLI for vision analysis."""
import argparse, base64, os
import openai
from ultralytics import YOLO

def analyze_image(image_path: str, question: str = None, model_path: str = "yolov8n.pt"):
    yolo = YOLO(model_path)
    results = yolo(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({"class": r.names[int(box.cls)], "confidence": float(box.conf), "bbox": box.xyxy[0].tolist()})
    print(f"Detected {len(detections)} objects: {[d['class'] for d in detections]}")
    if question:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        context = f"Objects detected: {', '.join(set(d['class'] for d in detections))}"
        resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":[
            {"type":"text","text":f"Context: {context}\n\nQuestion: {question}"},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
        ]}])
        print(f"Answer: {resp.choices[0].message.content}")
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default=None)
    args = parser.parse_args()
    analyze_image(args.image, args.question)
