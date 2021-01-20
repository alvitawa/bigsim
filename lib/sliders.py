import pygame
from pygame_widgets import Slider, TextBox

class LabeledSlider:

    def __init__(self, screen, pos_x, pos_y, label_text, width=200, label_size=150, min=0.0, max=1.0, initial=0.5, margin=20):
        self.width = width
        self.label_size = label_size

        self.label = TextBox(screen, pos_x, pos_y, label_size, 30, fontSize=16)
        self.label.setText(label_text)
        self.slider = Slider(screen, pos_x+self.label_size+margin, pos_y, self.width, 20, min=min, max=max, step=0.01, initial=initial)
        self.output = TextBox(screen, pos_x+self.label_size+self.width+margin*2, pos_y, 30, 20, fontSize=10)

    def draw(self):
        self.label.draw()
        self.slider.draw()
        self.output.draw()

    def update(self, events):
        self.slider.listen(events)
        self.output.setText("%.2f" % self.slider.getValue())

    def get_value(self):
        return self.slider.getValue()

    def set_value(self, value):
        self.slider.setValue(value)