from segment_anything_hq import SamPredictor, sam_model_registry
import numpy as np
import cv2

sam_checkpoint = "pretrained_checkpoint/sam_hq_vit_l.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

predictor = SamPredictor(sam)


def segment(img, bbox):
    
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)
    input_box = np.array(bbox)
    print(input_box, input_box.shape)
    input_point, input_label = None, None
    masks = []
    for box in input_box:
        mask, _, _ = predictor.predict(point_coords=input_point,
                                        point_labels=input_label,
                                        box = box,
                                        multimask_output=False,
                                        hq_token_only= False)
        masks.append(mask)
    return masks

def enhance_features(image, mask):
    
    image = cv2.imread(image)
    # Create a copy of the original image for processing
    enhanced_region = image.copy()
    
    # enhanced_region = cv2.GaussianBlur(enhanced_region, (5,5), cv2.BORDER_DEFAULT) 

    # Apply sharpening filter
    kernel = np.array([[-1, -1, -1], [-1, 7, -1], [-1, -1, -1]]) 
    enhanced_region = cv2.filter2D(enhanced_region, -1, kernel)
    
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = np.uint8(mask)

    # Apply the mask to extract the enhanced region
    enhanced_region = cv2.bitwise_and(enhanced_region, enhanced_region, mask=mask)

    return enhanced_region
    
def blend_images(image, enhanced_region, mask):
    
    image = cv2.imread(image)
    
    # Create a mask for the background
    background_mask = cv2.bitwise_not(mask)

    # Resize the background mask to match the image size
    background_mask = cv2.resize(background_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Blend the enhanced region with the background
    blended_image = cv2.bitwise_and(image, image, mask=background_mask)
    blended_image = cv2.bitwise_or(blended_image, enhanced_region)

    return blended_image