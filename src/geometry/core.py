from dataclasses import dataclass
import shapely


@dataclass
class Ellipse:
    """
    Класс эллипса
    """
    x: float
    y: float
    rx: float
    ry: float
    angle: float
    width: float
    height: float
    label: str

    @property
    def polygon(self):
        center = shapely.Point(self.x, self.y)
        circle = center.buffer(1.0, quad_segs=360)
        ellipse = shapely.affinity.scale(circle, self.rx, self.ry)
        return ellipse
