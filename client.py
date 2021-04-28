import requests
import json
import cv2

url = 'http://localhost:5000/predict'

files = {'snake': open('src/oiddata/test/random/snake.jpg', 'rb')}

response = requests.post(url, files=files)

res = json.loads(response.text)
print(res)

output_img = cv2.imread('src/oiddata/test/random/snake.jpg')
category = res[0]['label']
pr = res[0]['prob']
xmin, ymin, xmax, ymax = int(res[0]['bbox'][0]), int(res[0]['bbox'][1]), int(res[0]['bbox'][2]), int(res[0]['bbox'][3])
color = (255, 0, 0)

cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
cv2.putText(output_img, category + " : %.2f" % pr, (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            (255, 255, 255), 1)
cv2.imshow('prediction', output_img)
cv2.waitKey(0)
