


class objectview(object):
    def __init__(self, d) -> None:
        self.__dict__ = d
    def setattr(self, attr_name, attr_value):
        self.__dict__[attr_name] = attr_value