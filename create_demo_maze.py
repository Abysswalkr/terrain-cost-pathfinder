from PIL import Image, ImageDraw
import numpy as np

def create_demo_maze():
    tile_size = 10
    img = Image.new('RGB', (100, 100), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    def draw_tile(r, c, color):
        draw.rectangle([c*tile_size, r*tile_size, (c+1)*tile_size-1, (r+1)*tile_size-1], fill=color)

    draw_tile(2, 2, (255, 0, 0))
    draw_tile(2, 8, (0, 255, 0))
    
    for c in range(3, 8):
        draw_tile(2, c, (0, 0, 255))
        
    for r in range(3, 7): draw_tile(r, 2, (128, 128, 128))
    for c in range(3, 9): 
        print(f'Drawing (6, {c})')
        draw_tile(6, c, (128, 128, 128))
    for r in range(3, 6): draw_tile(r, 8, (128, 128, 128))
    
    img.save("demo_maze.bmp")
    
    # Verify directly:
    img_reload = Image.open('demo_maze.bmp').convert('RGB')
    tile = img_reload.crop((80,60,90,70))
    print('Verify (6,8):', np.array(tile).mean(axis=(0,1)))

if __name__ == '__main__':
    create_demo_maze()
