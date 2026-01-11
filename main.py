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

# Affine matrices in the form:
# a    c    e
# b    d    f
# 0    0    1
# [[a, c], [b, d]] for multiplication (stretch, shear, etc.)
# [e, f] for translation
# Row 3, col 3 added to matrix such that translations are also scaled during matrix multiplication (eg. objet rotates, translation also rotates)
@dataclass(frozen = True)
class mat2D:
    # Default values (no scale, no translation)
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    # Matrix multiplication, called using @ operator (eg. mat1 @ mat2)
    def __matmul__(self, o: "mat2D") -> "mat2D": # Takes mat2D, returns mat2D
        return mat2D(
            a = self.a * o.a + self.c * o.b,
            b = self.b * o.a + self.d * o.b,
            c = self.a * o.c + self.c * o.d,
            d = self.b * o.c + self.d * o.d,
            e = self.a * o.e + self.c * o.f + self.e,
            f = self.b * o.e + self.d * o.f + self.f,
        )
    
    # Apply matrix transformation to point
    def apply(self, p: point) -> point:
        x = p[0]
        y = p[2]

        # Transformation + translation
        # [[a, c][b, d]] * [x, y] + [e, f]
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)
    
    # Set e and f
    @staticmethod
    def translate(tx: float, ty: float) -> "mat2D":
        return mat2D(e = tx, f = ty)
    
    # Set a and d, set a = d if only one parameter given
    @staticmethod
    def scale(sx: float, sy: Optional[float] = None) -> "mat2D":
        if(sy is None):
            sy = sx
        return mat2D(a = sx, d = sy)

    # Rotate ccw
    @staticmethod
    def rotate(rad: float) -> "mat2D":
        cs = math.cos(rad)
        sn = math.sin(rad)
        return mat2D(a = cs, b = sn, c = -sn, d = cs)