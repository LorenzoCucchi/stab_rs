import stab_rs


class Geometry:
    length: float
    diameter: float
    area: float
    in_x: float
    in_y: float
    xcg: float
    mass: float
    mass_grain: float

    def __init__(length: float, diameter: float, in_x: float, in_y: float,
                 xcg: float, mass_grain: float) -> Geometry: ...

    @property.setter
    def set_length(self, length: float): ...

    @property.getter
    def get_length(self) -> float: ...

    @length.setter
    def set_length(self, length: float): ...

    @length.getter
    def get_length(self) -> float: ...

    @diameter.setter
    def set_diam(self, diameter: float): ...

    @diameter.getter
    def get_diam(self) -> float: ...

    @in_x.setter
    def set_inx(self, inx: float): ...

    @in_x.getter
    def get_inx(self) -> float: ...

    @in_y.setter
    def set_iny(self, iny: float): ...

    @in_y.getter
    def get_iny(self) -> float: ...

    @xcg.setter
    def set_xcg(self, xcg: float): ...

    @xcg.getter
    def get_xcg(self) -> float: ...

    @mass.setter
    def set_mass(self, mass: float): ...

    @mass.getter
    def get_mass(self) -> float: ...

    @mass_grain.setter
    def set_massgrain(self, mass_grain: float): ...

    @mass_grain.getter
    def get_massgrain(self) -> float: ...

    def print(self): ...
