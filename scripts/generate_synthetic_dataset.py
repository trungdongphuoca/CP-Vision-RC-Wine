import os
import random
from PIL import Image, ImageDraw, ImageFont

# Define YOLO classes
CLASSES = {
    "brand": 0,
    "vintage": 1,
    "variety": 2,
    "region": 3
}

# Sample data pools for synthetic generation
BRANDS = [
    "Chateau Margaux", "Opus One", "Penfolds", "Yellow Tail", 
    "Screaming Eagle", "Domaine de la Romanee-Conti", "Barefoot", 
    "Robert Mondavi", "Concha y Toro", "Antinori", "Santa Rita",
    "Villa Maria", "Casillero del Diablo", "Henschke", "Louis Latour"
]

VARIETIES = [
    "Cabernet Sauvignon", "Merlot", "Pinot Noir", "Chardonnay",
    "Sauvignon Blanc", "Syrah", "Shiraz", "Zinfandel", "Malbec",
    "Pinot Grigio", "Riesling", "Sangiovese", "Tempranillo"
]

REGIONS = [
    "Bordeaux", "Napa Valley", "Tuscany", "Mendoza", "Barossa Valley",
    "Burgundy", "Rioja", "Piedmont", "Champagne", "Marlborough",
    "Sonoma Coast", "Willamette Valley", "Chianti Classico"
]

# Standard Windows fonts that look somewhat like wine fonts
FONTS_AVAILABLE = ["times.ttf", "georgia.ttf", "arial.ttf", "calibri.ttf"]

def create_directories():
    os.makedirs("synthetic_dataset/images", exist_ok=True)
    os.makedirs("synthetic_dataset/labels", exist_ok=True)

def get_font(font_name, size):
    try:
        return ImageFont.truetype(font_name, size)
    except IOError:
        # Fallback if font is missing
        return ImageFont.load_default()

def yolo_format(bbox, img_width, img_height):
    # bbox is [left, top, right, bottom]
    x_center = ((bbox[0] + bbox[2]) / 2) / img_width
    y_center = ((bbox[1] + bbox[3]) / 2) / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def generate_label(index):
    # Create a blank label background (cream/white/off-white)
    bg_colors = [(253, 245, 230), (255, 255, 255), (245, 245, 220), (240, 248, 255)]
    img_width, img_height = random.randint(400, 600), random.randint(500, 800)
    img = Image.new('RGB', (img_width, img_height), color=random.choice(bg_colors))
    draw = ImageDraw.Draw(img)
    
    annotations = []
    
    # 1. Generate Brand (Usually at the top or middle, large font)
    brand = random.choice(BRANDS)
    font_brand = get_font(random.choice(["times.ttf", "georgia.ttf"]), random.randint(40, 60))
    # Get bounding box of text
    left, top, right, bottom = draw.textbbox((0, 0), brand, font=font_brand)
    tw, th = right - left, bottom - top
    tx, ty = (img_width - tw) // 2, random.randint(50, 150)
    draw.text((tx, ty), brand, fill=(0, 0, 0), font=font_brand)
    # add padding to bbox
    annotations.append((CLASSES["brand"], [tx, ty, tx+tw, ty+th]))
    
    # 2. Generate Vintage (Usually year, medium font)
    vintage = str(random.randint(1990, 2024))
    font_vintage = get_font(random.choice(["arial.ttf", "times.ttf"]), random.randint(30, 45))
    left, top, right, bottom = draw.textbbox((0, 0), vintage, font=font_vintage)
    tw, th = right - left, bottom - top
    tx, ty = (img_width - tw) // 2, ty + random.randint(80, 120)
    draw.text((tx, ty), vintage, fill=(50, 0, 0), font=font_vintage)
    annotations.append((CLASSES["vintage"], [tx, ty, tx+tw, ty+th]))
    
    # 3. Generate Variety
    variety = random.choice(VARIETIES)
    font_variety = get_font(random.choice(["georgia.ttf", "times.ttf"]), random.randint(35, 50))
    left, top, right, bottom = draw.textbbox((0, 0), variety, font=font_variety)
    tw, th = right - left, bottom - top
    tx, ty = (img_width - tw) // 2, ty + random.randint(80, 120)
    draw.text((tx, ty), variety, fill=(0, 0, 0), font=font_variety)
    annotations.append((CLASSES["variety"], [tx, ty, tx+tw, ty+th]))
    
    # 4. Generate Region (Usually bottom, smaller font)
    region = random.choice(REGIONS)
    font_region = get_font(random.choice(["arial.ttf", "calibri.ttf"]), random.randint(25, 35))
    left, top, right, bottom = draw.textbbox((0, 0), region, font=font_region)
    tw, th = right - left, bottom - top
    tx, ty = (img_width - tw) // 2, ty + random.randint(80, 150)
    draw.text((tx, ty), region, fill=(100, 100, 100), font=font_region)
    annotations.append((CLASSES["region"], [tx, ty, tx+tw, ty+th]))
    
    # Save image
    img_filename = f"synthetic_dataset/images/label_{index:04d}.jpg"
    img.save(img_filename)
    
    # Save YOLO annotations
    txt_filename = f"synthetic_dataset/labels/label_{index:04d}.txt"
    with open(txt_filename, "w") as f:
        for class_id, bbox in annotations:
            yolo_box = yolo_format(bbox, img_width, img_height)
            f.write(f"{class_id} {yolo_box}\n")

if __name__ == "__main__":
    print("Generating Synthetic Wine Label Dataset for YOLOv8...")
    create_directories()
    
    # Create classes.txt
    with open("synthetic_dataset/classes.txt", "w") as f:
        for name in sorted(CLASSES.keys(), key=lambda k: CLASSES[k]):
            f.write(f"{name}\n")
            
    num_samples = 100
    for i in range(num_samples):
        generate_label(i)
        if (i+1) % 20 == 0:
            print(f"Generated {i+1}/{num_samples} samples...")
            
    print("Generation complete! Dataset saved in 'synthetic_dataset/'.")
    print("You can now use this dataset to train YOLOv8 to detect Brand, Vintage, Variety, and Region directly!")
