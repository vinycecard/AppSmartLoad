from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

class MainWindow(Screen):
    pass

class CalibrateWindow(Screen):
    pass

class StartWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("smartLoad.kv")

class SmartLoadApp(App):
    def build(self):
        return kv

if __name__ == "__main__":
    SmartLoadApp().run()