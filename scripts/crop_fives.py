from PIL import Image
import os

def crop_image_to_tiles(img_path, output_dir, tile_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(img_path)
    img_width, img_height = img.size
    tile_width, tile_height = tile_size

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    count = 0
    for top in range(0, img_height, tile_height):
        for left in range(0, img_width, tile_width):
            right = min(left + tile_width, img_width)
            bottom = min(top + tile_height, img_height)
            tile = img.crop((left, top, right, bottom))
            tile_name = f"{base_name}_tile_{count}.png"
            tile.save(os.path.join(output_dir, tile_name), format="PNG")
            count += 1
    print(f"{base_name} done: {count} tiles saved.")

def batch_crop_folder(input_dir, output_dir, tile_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    for i, filename in enumerate(files):
        print(f'{i+1}/{len(files)}')
        img_path = os.path.join(input_dir, filename)
        crop_image_to_tiles(img_path, output_dir, tile_size)

# 示例调用
batch_crop_folder(
    input_dir='../datasets/FIVES/test/Original',         # 输入图像文件夹
    output_dir='../datasets/FIVES/512/test/Original',  # 输出 tile 保存目录
    tile_size=(512, 512)        # 切片尺寸
)
