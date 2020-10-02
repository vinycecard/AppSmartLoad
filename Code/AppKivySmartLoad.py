import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

class SmartLoadGrid(GridLayout):
    def __init__(self, **kwargs):
        super(SmartLoadGrid, self).__init__(**kwargs)
        self.cols = 1

        self.inside1 = GridLayout()
        self.inside1.cols = 1
        self.inside1.add_widget(Label(text="App Smart Load", font_size=40))
        self.add_widget(self.inside1)

        self.inside2 = GridLayout()
        self.inside2.cols = 2
        self.inside2.add_widget(Label(text="Load [Kg]: ", font_size=20))
        self.load = TextInput(multiline=False, font_size=20)
        self.inside2.add_widget(self.load)
        self.inside2.add_widget(Label(text="Name: ", font_size=20))
        self.name = TextInput(multiline=False, font_size=20)
        self.inside2.add_widget(self.name)
        self.inside2.add_widget(Label(text="Email: ", font_size=20))
        self.email = TextInput(multiline=False,font_size=20)
        self.inside2.add_widget(self.email)
        self.add_widget(self.inside2)
 
        self.inside3 = GridLayout()
        self.inside3.cols = 2
        self.inside3.submitCalibrate = Button(text="Calibrate", font_size=40, color = (0.3,0.6,0.7,1),
        background_color = (0.3,.4,.5,1))
        self.inside3.submitCalibrate.bind(on_press=self.pressed)
        self.inside3.add_widget(self.inside3.submitCalibrate)       
        self.inside3.submitStart = Button(text="Start", font_size=40,  color = (0.3,0.6,0.7,1),
        background_color = (0.3,.4,.5,1))
        self.inside3.submitStart.bind(on_press=self.pressed)
        self.inside3.add_widget(self.inside3.submitStart)
        self.add_widget(self.inside3)       

    def pressed(self, instance):
        name = self.name.text
        load = self.load.text
        email = self.email.text

        print("Load:", load, "Name:", name, "Email:", email)


class SmartLoadApp(App):
    def build(self):
        return SmartLoadGrid()

if __name__ == "__main__":
    SmartLoadApp().run()