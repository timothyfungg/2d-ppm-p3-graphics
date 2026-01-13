import math
from dataclasses import dataclass

Colour = tuple[int, int, int] # In the form [R, G, B]
Point = tuple[float, float] # In the form [x, y]

class Framebuffer:
    def __init__(self, w: int, h: int, bg: Colour = (255, 255, 255)):
        self.w, self.h = w, h
        # Initialize screen with w columns and h rows as a bg colour background
        self.pixels = [[bg for _ in range(w)] for _ in range(h)] # to access each px, [y][x] starting at (0, 0)
    
    def set_px(self, x: int, y: int, colour: Colour):
        if(0 <= x < self.w and 0 <= y < self.h):
            self.pixels[y][x] = colour

    def save_ppm(self, path: str):
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
class Mat2D:
    # Default values (no scale, no translation)
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    # Matrix multiplication, called using @ operator (eg. mat1 @ mat2)
    def __matmul__(self, o: "Mat2D") -> "Mat2D": # Takes Mat2D, returns Mat2D
        return Mat2D(
            a = self.a * o.a + self.c * o.b,
            b = self.b * o.a + self.d * o.b,
            c = self.a * o.c + self.c * o.d,
            d = self.b * o.c + self.d * o.d,
            e = self.a * o.e + self.c * o.f + self.e,
            f = self.b * o.e + self.d * o.f + self.f,
        )
    
    # Apply matrix transformation to point
    def apply(self, p: Point) -> Point:
        x, y = p

        # Transformation + translation
        # [[a, c][b, d]] * [x, y] + [e, f]
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)
    
    # Set e and f
    @staticmethod
    def translate(tx: float, ty: float) -> "Mat2D":
        return Mat2D(e = tx, f = ty)
    
    # Set a and d, set a = d if only one parameter given
    @staticmethod
    def scale(sx: float, sy: float | None = None) -> "Mat2D":
        if(sy is None):
            sy = sx
        return Mat2D(a = sx, d = sy)

    # Rotate ccw
    @staticmethod
    def rotate(rad: float) -> "Mat2D":
        cs, sn = math.cos(rad), math.sin(rad)
        return Mat2D(a = cs, b = sn, c = -sn, d = cs)

# Draws line between two points using Bresenham's line algorithm
def draw_line(fb: Framebuffer, p0: Point, p1: Point, colour: Colour) -> None:
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
# Can visualize with right hand rule on AB x AC
# Run 3 times for each pair of vertices to determine if c is within the triangle (if so, all 3 results will be the same sign)
# If all signs are the same, that means c is to the left/right of all vertices and is therefore inside the triangle
def edge_fn(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def fill_triangle(fb: Framebuffer, a: Point, b: Point, c: Point, colour: Colour) -> None:
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
        b, c = c, b  # lip

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
# Process (using cartesian plane)
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
def is_ccw(poly: list[Point]) -> bool:
    s = 0.0 # Sum of area
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)] # Mod length resets the index to 0 (returns to start)
        s += (x2 - x1) * (y2 + y1) # Calculate area, add to sum
    return s < 0

# Uses edge_fn to check if point is within triangle
def point_in_tri(p: Point, a: Point, b: Point, c: Point) -> bool:
    # Check all pairs of vertices
    w0 = edge_fn(a, b, p)
    w1 = edge_fn(b, c, p)
    w2 = edge_fn(c, a, p)

    has_neg = (w0 < 0) or (w1 < 0) or (w2 < 0) # True if any area is negative
    has_pos = (w0 > 0) or (w1 > 0) or (w2 > 0) # True if any area is positive
    return not (has_neg and has_pos) # True if either no negative or no positive

# Tessellate polygon into triangles, takes list of points, returns list of triangles
# Process
# 1. Check if corner is convex by using edge_fn on a side (prev cur) and the next vertex (nxt)
#    Convex if positive
# 2. Check if ther are no other vertices in the corner by using point_in_tri
# 3. If 1 and 2 are true, cut triangle
# 4. Continue to next corner
def triangulate_ear_clipping(poly: list[Point]) -> list[tuple[Point, Point, Point]]:
    # Simple ear clipping for simple polygons (no self-intersections).
    if len(poly) < 3:
        return []
    pts = poly[:] # Shallow copy of poly (deep copy not necessary since not mutating any inner wrappers, only removing them from list)

    # Check for ccw such that only all positive case must be checked
    if not is_ccw(pts):
        pts.reverse()

    triangles: list[tuple[Point, Point, Point]] = [] # Output

    # Check if polygon is convex
    def is_convex(prev: Point, cur: Point, nxt: Point) -> bool:
        return edge_fn(prev, cur, nxt) > 0  # CCW convex

    i = 0
    guard = 0 # Exit loop if stuck
    while len(pts) > 3 and guard < 10000:
        guard += 1
        
        # Sliding window
        # Mod by length to loop back to beginning
        prev = pts[(i - 1) % len(pts)]
        cur  = pts[i % len(pts)] # Corner being cut off
        nxt  = pts[(i + 1) % len(pts)]

        if is_convex(prev, cur, nxt):
            # Check for other points within ear
            ear_ok = True
            for p in pts:
                # Skip to next loop if checking ear vertex (save time)
                if p in (prev, cur, nxt):
                    continue
                # Check if point is in triangle, terminate if so
                if point_in_tri(p, prev, cur, nxt):
                    ear_ok = False
                    break
            # If both checks succeed, add triangle to output list and remove corner from list and reset loop
            if ear_ok:
                triangles.append((prev, cur, nxt))
                pts.pop(i % len(pts))
                i = 0
                continue
        
        # Increment by 1, loop back to start
        i = (i + 1) % len(pts)

    # If polygon is a triangle
    if len(pts) == 3:
        triangles.append((pts[0], pts[1], pts[2]))
    return triangles

# Outline of shape
@dataclass
class Stroke:
    colour: Colour = (0, 0, 0)

# Fill of shape
@dataclass
class Fill:
    colour: Colour | None = None

# Parent class of all objects
class Node:
    # Optional affine matrix
    def __init__(self, transform: Mat2D | None = None):
        self.transform = transform or Mat2D() # Default affine matrix if none given

    # Force all children to have this method
    def render(self, fb: Framebuffer, parent_tf: Mat2D) -> None:
        raise NotImplementedError

# Child of Node, holds other objects/children
# Allows for creation of complex objects using smaller objects
class Group(Node):
    # Optional list of children and affine matrix
    def __init__(self, children: list[Node] | None = None, transform: Mat2D | None = None):
        super().__init__(transform)
        self.children: list[Node] = children or []

    # Loops through children and applies affine transformation
    def render(self, fb: Framebuffer, parent_tf: Mat2D) -> None:
        tf = parent_tf @ self.transform # Applies new transformation to self
        for ch in self.children:
            ch.render(fb, tf) # Calls the render function of the child

# Child of Node, represents a line
class Line(Node):
    # Takes two points, a stroke colour and an optional affine matrix
    def __init__(self, p0: Point, p1: Point, stroke = Stroke(), transform: Mat2D | None = None):
        super().__init__(transform)
        self.p0, self.p1 = p0, p1
        self.stroke = stroke

    # Draws line with transformed points and stroke colour
    def render(self, fb: Framebuffer, parent_tf: Mat2D) -> None:
        tf = parent_tf @ self.transform
        draw_line(fb, tf.apply(self.p0), tf.apply(self.p1), self.stroke.colour)

# Child of Node, represents a polygon
class Polygon(Node):
    # Takes a list of vertices, stroke colour, optional fill colour and optional affine matrix
    def __init__(self, pts: list[Point], stroke = Stroke(), fill = Fill(None), transform: Mat2D | None = None):
        super().__init__(transform)
        self.pts = pts
        self.stroke = stroke
        self.fill = fill

    # Draws polygon with transformed points, stroke colour and fill colour
    def render(self, fb: Framebuffer, parent_tf: Mat2D) -> None:
        tf = parent_tf @ self.transform
        pts = [tf.apply(p) for p in self.pts] # Apply transformation to all points

        # If there is a fill colour and polygon has volume
        if self.fill.colour is not None and len(pts) >= 3:
            tris = triangulate_ear_clipping(pts) # Triangulate polygon
            # Loop through triangle list and fill each triangle
            for a, b, c in tris:
                fill_triangle(fb, a, b, c, self.fill.colour)

        # If polygon has no volume, draw a line
        if len(pts) >= 2:
            for i in range(len(pts)):
                draw_line(fb, pts[i], pts[(i + 1) % len(pts)], self.stroke.colour)