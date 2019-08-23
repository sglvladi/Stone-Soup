import zeep
import requests
import time
from onvif import ONVIFCamera, ONVIFService
from .tcpsocket import TcpClient

def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue


zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue


class Onvif_PTZ_Camera:

    def __init__(self, ip, port, username, password):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self._camera = ONVIFCamera(
            ip, port, username, password)
        self.profile = self._camera.create_media_service().GetProfiles()[0]
        self.ptz_service = self._camera.create_ptz_service()

        request = self.ptz_service.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self.profile.PTZConfiguration.token

        self.ptz_config_options = self.ptz_service.GetConfigurationOptions(
            request)

        self.speed_limits = {
            'Pan': {
                'MIN': self.ptz_config_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min,
                'MAX': self.ptz_config_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
            },
            'Tilt': {
                'MIN': self.ptz_config_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min,
                'MAX': self.ptz_config_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
            },
            'Zoom': {
                'MIN': self.ptz_config_options.Spaces.ContinuousZoomVelocitySpace[0].XRange.Min,
                'MAX': self.ptz_config_options.Spaces.ContinuousZoomVelocitySpace[0].XRange.Max
            }
        }

    def continuous_move(self, pan_speed=0, tilt_speed=0, zoom_speed=0):

        moverequest = self.ptz_service.create_type('ContinuousMove')
        moverequest.ProfileToken = self.profile.token
        if moverequest.Velocity is None:
            moverequest.Velocity = self.ptz_service.GetStatus(
                {'ProfileToken': self.profile.token}).Position

        moverequest.Velocity.PanTilt.x = self.speed_limits["Pan"]["MAX"]*pan_speed
        moverequest.Velocity.PanTilt.y = self.speed_limits["Tilt"]["MAX"]*tilt_speed
        moverequest.Velocity.Zoom.x = self.speed_limits["Zoom"]["MAX"]*zoom_speed

        self.ptz_service.ContinuousMove(moverequest)

    def move_http(self, velocity=(1, 1)):
        panSpeed = round(velocity[0]*8)
        tiltSpeed = round(velocity[1]*8)

        command = 0  # Stop command
        if panSpeed == 0:
            # Up or down
            if(tiltSpeed < 0):
                # Up
                command = 2
            elif(tiltSpeed > 0):
                # Down
                command = 1
        elif tiltSpeed == 0:
            # Left or Right
            if(panSpeed < 0):
                # Left
                command = 3
            elif(panSpeed > 0):
                # Right
                command = 4
        else:
            if(panSpeed < 0 and tiltSpeed < 0):
                # Down left
                command = 5
            elif(panSpeed < 0 and tiltSpeed > 0):
                # Up left
                command = 6
            elif(panSpeed > 0 and tiltSpeed > 0):
                # Up Right
                command = 7
            elif(panSpeed > 0 and tiltSpeed < 0):
                # Down Right
                command = 8

        url = "http://192.168.0.10/form/setPTZCfg"
        data = {"command": command,
                "ZFSpeed": 0, "PTSpeed": 0,
                "panSpeed": abs(panSpeed), "tiltSpeed": abs(tiltSpeed),
                "focusSpeed": 2, "zoomSpeed": 2}
        return requests.post(url, data=data, auth=(self.username, self.password))

    def zoom_http(self, velocity=1):
        command = 0
        if velocity > 0:
            command = 13
        elif velocity < 0:
            command = 14

        url = "http://192.168.0.10/form/setPTZCfg"
        data = {"command": command,
                "ZFSpeed": 0, "PTSpeed": 0,
                "panSpeed": 0, "tiltSpeed": 0,
                "focusSpeed": 2, "zoomSpeed": 2}
        return requests.post(url, data=data, auth=(self.username, self.password))

    def freeze(self):

        moverequest = self.ptz_service.create_type('ContinuousMove')
        moverequest.ProfileToken = self.profile.token
        if moverequest.Velocity is None:
            moverequest.Velocity = self.ptz_service.GetStatus(
                {'ProfileToken': self.profile.token}).Position

        moverequest.Velocity.PanTilt.x = 0
        moverequest.Velocity.PanTilt.y = 0

        self.ptz_service.ContinuousMove(moverequest)

    def getPtzStatus(self):
        status = self.ptz_service.GetStatus(
            {'ProfileToken': self.profile.token})
        return status

    def getPtzConfiguration(self):
        return self.ptz_service.GetConfigurations()

    def moveAbsolute(self):
        status = self.ptz_service.GetStatus(
            {'ProfileToken': self.profile.token})
        moverequest = self.ptz_service.create_type('AbsoluteMove')
        moverequest.ProfileToken = self.profile.token
        moverequest.Position = status.Position
        moverequest.Position.x = 0.5
        moverequest.Position.y = 0.5
        self.ptz_service.AbsoluteMove(moverequest)


class DenbridgeCctvCamera(TcpClient):

    def __init__(self, target, port):
        super().__init__(target, port)

class FloureonCctvCamera(Onvif_PTZ_Camera):

    def send_request(self):
        feedback = {'fov': 0, 'pan': 0, 'panSpeed': 0, 'tilt': 0,
                    'tiltSpeed': 0, 'timestamp': '2019-08-22T22:45:36',
                    'zoom': 0}
        return feedback

    def send_command(self, command):
        panSpeed, tiltSpeed = (command["panSpeed"],
                               command["tiltSpeed"])
        zoomSpeed = command["zoomSpeed"]

        if panSpeed + tiltSpeed == 0:
            if zoomSpeed != 0:
                # Control the zoom
                self.zoom_http(zoomSpeed)
                time.sleep(abs(zoomSpeed))
                self.zoom_http(0)
            else:
                self.freeze()
        else:
            # Control the pan-tilt
            self.move_http((panSpeed, tiltSpeed))
            #self.continuous_move(panSpeed, tiltSpeed)
            converged = False

        return True
