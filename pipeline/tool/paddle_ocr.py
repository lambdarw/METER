from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

def pp_ocr(img_path, ocr_model):
    print('ppocr')
    ocr_result = ocr_model.ocr(img_path, cls=True)
    print('ppocr_finish')
    print(ocr_result)

    boxes = []
    phrases = []
    if ocr_result[0]:
        for res in ocr_result[0]:
            box = [tuple(point) for point in res[0]]
            box = [min(point[0] for point in box), min(point[1] for point in box),
                        max(point[0] for point in box), max(point[1] for point in box)]
            boxes.append(box)
            
            phrases.append(res[-1][0])

    ocr = {
        'boxes': boxes,
        'phrases': phrases
    }

    return ocr