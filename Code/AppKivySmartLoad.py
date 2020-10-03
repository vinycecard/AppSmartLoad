from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.label import Label

class MainWindow(Screen):
    username = ObjectProperty(None)
    email = ObjectProperty(None)
    load = ObjectProperty(None)

    def isValidLoad(self):
        if (self.load.text != '' and float(self.load.text) > 0):
            return True
        else:  
            self.invalidLoadPopup()
            self.load.text = ''
            return False

    def invalidLoadPopup(self):

            pop = Popup(title='Invalid Load',
                  content=Label(text='The load must be a value greater than 0'),
                  size_hint=(None, None), size=(350, 350) )
            pop.open()
    

class CalibrateWindow(Screen):
    pass

class StartWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass
    

kv = Builder.load_file("smartLoad.kv")

screenmanager = WindowManager()

screens = [MainWindow(name="main"), StartWindow(name="start"), CalibrateWindow(name="calibrate")]
for screen in screens:
    screenmanager.add_widget(screen)

screenmanager.current = "main"

class SmartLoadApp(App):
    def build(self):
        return screenmanager

if __name__ == "__main__":
    SmartLoadApp().run()