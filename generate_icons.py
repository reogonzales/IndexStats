from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("static/icons", exist_ok=True)

for size in (192, 512):
    img = Image.new("RGB", (size, size), color="#2196F3")
    draw = ImageDraw.Draw(img)
    font_size = size // 5
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    text = "IX"
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (size - (bbox[2] - bbox[0])) // 2
    y = (size - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill="white", font=font)
    img.save(f"static/icons/icon-{size}.png")
    print(f"Created static/icons/icon-{size}.png")
