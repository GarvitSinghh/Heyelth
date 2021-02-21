from plyer import notification
import os

def notify(title, description):

    icon_path = os.path.abspath('./eyeIcon.ico')

    notif = notification.notify(title=title, message=description, app_icon=icon_path, timeout=5)
