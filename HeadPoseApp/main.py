from headposecamera import HeadPoseCamera

from kivy.utils            import platform
from kivy.uix.boxlayout    import BoxLayout
from kivy.app              import App
from kivy.lang             import Builder

class CameraLayout(BoxLayout):
    def __init__(self, **kwargs):
        self._request_android_permissions()
        super(CameraLayout, self).__init__(**kwargs)

    @staticmethod
    def is_android():
        return platform == 'android'

    def _request_android_permissions(self):
        """
        requests camera permissions
        """
        if not self.is_android():
            return
        from android.permissions import request_permission, Permission
        request_permission(Permission.CAMERA)


Builder.load_string('''

<CameraLayout>:
    size: root.size
    orientation: 'vertical'
    HeadPoseCamera:
        id: camera
        resoltuion: (640, 480)
        size_hint: 1, .7
        play: True

''')

class HeadPoseApp(App):
    def build(self):
        self.cameralayout = CameraLayout()
        return self.cameralayout
    def on_stop(self):
        cam = self.cameralayout.ids['camera']
        cam._camera.stop()

HeadPoseApp().run()
