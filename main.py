import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

colour = Tuple[int, int, int] # In the form [R, G, B]
point = Tuple[float, float] # In the form [x, y]

class Framebuffer:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        bg = (255, 255, 255) # White
        # Initialize screen with w columns and h rows as a white background
        self.pixels = [[bg for _ in range(w)] for _ in range(h)] # to access each px, [y][x] starting at (0, 0)
    
    def setPx(self, x: int, y: int, colour: colour):
        if(0 <= y < self.y and 0 <= x < self.w):
            self.pixels[y][x] = colour

    def savePpm(self, path: str):
        with open(path, "w") as f:
            # Write header:
                # P3
                # Width Height
                # 255 (max value)
            f.write(f"P3\n{self.w} {self.h}\n255\n")

            # For each row, write each pixel's RGB values to the line using generators
            for row in self.pixels:
                f.write(" ".join(f"{r} {g} {b}" for (r, g, b) in row)) # Loop through each pixel (tuple) and create a string (eg. '255, 255, 255') that is then joined to the larger string
                f.write("\n")

