class Data:  # Klasa do przechowywania pojedynczej danej z pliku
    vector1 = [0, 0, 0, 0]
    label = ""

    def __init__(self, data):
        self.vector = [data[0], data[1], data[2], data[3]]
        self.label = data[4]
