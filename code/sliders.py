import pygame
from pygame_widgets import Slider, TextBox

class LabeledSlider:

    def __init__(self, screen, pos_x, pos_y, label_text, width=150, label_size=60):
        self.width = width
        self.label_size = label_size

        self.label = TextBox(screen, pos_x, pos_y, self.label_size, 20, fontSize=10)
        self.label.setText("Cohesion")
        self.slider = Slider(screen, pos_x+self.label_size, pos_y, self.width, 20, min=0.0, max=1.0, step=0.01)
        self.output = TextBox(screen, pos_x+self.label_size+self.width, pos_y, 30, 20, fontSize=10)

    def draw(self):
        self.label.draw()
        self.slider.draw()
        self.output.draw()

    def update(self, events):
        self.slider.listen(events)
        self.output.setText("%.2f" % self.slider.getValue())

    def get_value(self):
        self.slider.getValue()