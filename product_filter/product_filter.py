from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import cv2
import numpy as np
from product_segment import segment, enhance_features, blend_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")

def infer(img, text_queries, score_threshold):

    queries=""
    for query in text_queries:
        queries += f"{query}. "
    img = cv2.imread(img)
    width, height = img.shape[:2]

    target_sizes=[(width, height)]
    inputs = dino_processor(text=queries, images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)
        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu()
        results = dino_processor.post_process_grounded_object_detection(outputs=outputs, input_ids=inputs.input_ids,
                                                                        box_threshold=score_threshold,
                                                                        target_sizes=target_sizes)
    
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    result_labels = []
    result_bbox = []

    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        if score < score_threshold:
            continue
        
        if label != "":
            result_labels.append((box, label))
            result_bbox.append(box)
    print(result_labels)
    return result_labels, result_bbox      

def filter_product(img, text_queries, dino_threshold):
    # text_queries = text_queries
    # text_queries = text_queries.split(",")
    labels, bbox = infer(img, text_queries, dino_threshold)
    seg_mask = segment(img, bbox)
    if len(seg_mask) == 0:
        return None
    
    
    # Combine the masks using logical OR operation
    combined_mask = np.zeros_like(seg_mask[0][0], dtype=bool)
    for mask in seg_mask:
        combined_mask |= mask[0]

    combined_mask = combined_mask.astype(np.uint8)
    # cv2.imwrite('mask.png', combined_mask)
    
    enhanced_region = enhance_features(img, combined_mask)
    
    enhanced_image = blend_images(img, enhanced_region, combined_mask)
    
    return enhanced_image

# enhanced_image = filter_product('/home/ai-4/Downloads/archive (1)/fashion-dataset/fashion-dataset/images/2785.jpg',
#                   'Shoe, Sneaker, Bottle, Cup, Sandal, Perfume, Toy, Sunglasses, Car, Water Bottle, Chair, Office Chair, Can, Cap, Hat, Couch, Wristwatch, Glass, Bag, Handbag, Baggage, Suitcase, Headphones, Jar, Vase',
#                   0.3)

# cv2.imwrite('enhanced_image.png', enhanced_image)