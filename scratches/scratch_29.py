








class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return 6

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)

    def area(self):
        return 1


class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 47

class RightPyramid(Square,Triangle):
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height

    def area(self):
        base_area = super(Triangle,self).area()

        return base_area


pyramid = RightPyramid(2, 4)
print(pyramid.area())