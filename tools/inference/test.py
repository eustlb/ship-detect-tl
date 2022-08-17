bboxs = [(591.0, 740.0, 628.0, 768.0, 'ship'), (601.0, 732.0, 637.0, 762.0, 'ship'), (657.0, 710.0, 672.0, 717.0, 'ship'), (662.0, 715.0, 672.0, 720.0, 'ship'), (568.0, 661.0, 573.0, 665.0, 'ship'), (582.0, 629.0, 587.0, 638.0, 'ship'), (657.0, 705.0, 670.0, 712.0, 'ship'), (648.0, 708.0, 656.0, 713.0, 'ship'), (633.0, 755.0, 645.0, 767.0, 'ship')]

bboxs = [[el[1]/768, el[0]/768, el[3]/768, el[2]/768] for el in bboxs]
scores = [1. for i in bboxs]

from detections import save_image_bbox


save_image_bbox('/tf/test/results/test.png', '/tf/test/results/test1.png', bboxs, scores)


