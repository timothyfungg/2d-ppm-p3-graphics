import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

colour = Tuple[int, int, int] # In the form [R, G, B]
point = Tuple[float, float] # In the form [x, y]

class Framebuffer:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
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
        x, y = p

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
        cs, sn = math.cos(rad), math.sin(rad)
        return mat2D(a = cs, b = sn, c = -sn, d = cs)

# Draws line between two points using Bresenham's line algorithm
def draw_line(fb: Framebuffer, p0: point, p1: point, colour: colour) -> None:
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        fb.set_px(x0, y0, colour)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

# Takes 2 vertices of a triangle (a, b) and a point (c) and computes the cross product (same magnitude as determiant for 2x2) between AB and AC
# Run 3 times for each pair of vertices to determine if c is within the triangle (if so, all 3 results will be the same sign)
# If all signs are the same, that means c is to the left/right of all vertices and is therefore inside the triangle
def edge_fn(a: point, b: point, c: point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def fill_triangle(fb: Framebuffer, a: point, b: point, c: point, colour: colour) -> None:
    # Rectangle containing triangle, check these pixels
    minx = int(math.floor(min(a[0], b[0], c[0])))
    maxx = int(math.ceil (max(a[0], b[0], c[0])))
    miny = int(math.floor(min(a[1], b[1], c[1])))
    maxy = int(math.ceil (max(a[1], b[1], c[1])))

    area = edge_fn(a, b, c)
    if area == 0:
        return
    # Change AB x AC to AC x AB if negative, such that only all positives must be checked
    if area < 0:
        b, c = c, b  # flip

    # Loop through each row of pixels
    for y in range(miny, maxy + 1):
        # Loop through each pixel in the row
        for x in range(minx, maxx + 1):
            p = (x + 0.5, y + 0.5) # Ensures the center of the pixel is checked
            
            # Check vertices
            w0 = edge_fn(b, c, p)
            w1 = edge_fn(c, a, p)
            w2 = edge_fn(a, b, p)

            # Check if positive, set colour if so
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                fb.set_px(x, y, colour)

# Checks if list of vertices are progressing cw or ccw
# Process explanation (using cartesian plane)
# 1. Divide polygon up with vertical lines, creating trapezoids between x axis and top of polygon and x axis and bottom of polygon
# 2. Calculate area of each trapezoid: (x_2 - x_1)(y_2 + y_1) / 2, the half is dropped because only the sign matters
#    Formula is identical to [(a + b) * h] / 2, where h = c (right trapezoid) and c = x_2 - x_1 and a + b = y_1 + y_2
#    Or a and b are the parallel sides and c is the side perpendicular to the x-axis
# 3. Sum the area of each trapezoid
#    Magnitude of trapezoids calculated at the top are larger than the bottom, therefore if moving left at the top, delta x will be negative and area will be negative
#    If moving right at the top, delta x will be positive and area will be positive
#    Note: similar to a circle, when moving cw, the top half will always be completed moving right, no matter the starting position, the converse also holds true
# 4. Moving right at top (positive area): cw
#    Moving left at top (negative area): ccw
def is_ccw(poly: List[point]) -> bool:
    s = 0.0 # Sum of area
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)] # Mod length resets the index to 0 (returns to start)
        s += (x2 - x1) * (y2 + y1) # Calculate area, add to sum
    return s < 0

# Uses edge_fn to check if point is within triangle
def point_in_tri(p: point, a: point, b: point, c: point) -> bool:
    # Check all pairs of vertices
    w0 = edge_fn(a, b, p)
    w1 = edge_fn(b, c, p)
    w2 = edge_fn(c, a, p)

    has_neg = (w0 < 0) or (w1 < 0) or (w2 < 0) # True if any area is negative
    has_pos = (w0 > 0) or (w1 > 0) or (w2 > 0) # True if any area is positive
    return not (has_neg and has_pos) # True if either no negative or no positive