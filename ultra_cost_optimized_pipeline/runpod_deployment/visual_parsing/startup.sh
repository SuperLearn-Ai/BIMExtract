
# Install visual parsing dependencies
pip install --no-cache-dir paddlepaddle-gpu>=2.5.2 paddleocr>=2.7.3
pip install --no-cache-dir nougat-ocr>=0.1.17 layoutparser[layoutmodels,tesseract,ocr]>=0.3.4
pip install --no-cache-dir transformers>=4.36.0 torch>=2.1.0 opencv-python>=4.8.0
pip install --no-cache-dir Pillow>=10.0.0 runpod>=1.5.0

# Initialize models
python -c "
import paddleocr
from nougat import predict
from transformers import LayoutLMTokenizer

print('Initializing PaddleOCR...')
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

print('Downloading Nougat model...')
nougat_model = predict.download_model()

print('Downloading LayoutLM...')
layoutlm_tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')

print('All visual models initialized!')
"

python handler.py
                