import torch
import numpy as np
from omegaconf import OmegaConf
from src.models import EffDetWrapper
from src.transforms import get_inference_transforms

state_dict = torch.load('model_checkpoints/effdet_standard_2026-02-03_14.51.43/effdet_standard_2026-02-03_14.51.43_epoch_3_vloss-0.190992.pth')

conf = OmegaConf.load('model_checkpoints/effdet_standard_2026-02-03_14.51.43/effdet_standard_2026-02-03_14.51.43_config.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EffDetWrapper(conf=conf, device=device)
model.load_state_dict(state_dict['ema_state_dict'])
model.eval_mode()

print("Model state dict loaded successfully.")

transforms = get_inference_transforms(resize=conf.images.resize)

test_image_path = 'data/raw/1de986c8-4e41-40e1-a2a7-de5ddd97bd15.jpg'
import cv2
image = cv2.imread(test_image_path, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)
h, w, _ = image.shape
input_tensor = transforms(image=image)['image'].unsqueeze(0).to(device)
outputs = model.predict(input_tensor)

outputs = outputs.cpu().numpy()
top = outputs[np.where(outputs[:, :, 4] > 0.2)]

bboxes = top[:, :4].astype(np.int32)
scores = top[:, 4]
labels = top[:, 5].astype(np.int32)

out_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

for i, box in enumerate(bboxes):
    x1, y1, x2, y2 = box
    w_scale = w / conf.images.resize
    h_scale = h / conf.images.resize
    x1, x2 = (int(x1 * w_scale), int(x2 * w_scale))
    y1, y2 = (int(y1 * h_scale), int(y2 * h_scale))

    cv2.rectangle(out_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(out_image, f'{scores[i]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 3)
cv2.imwrite('output_detections.jpg', out_image)