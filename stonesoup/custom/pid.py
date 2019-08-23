import numpy as np
import time
import json
from datetime import datetime, timedelta


class FlirPidController:
    error = []
    error_z = []
    sat_pt = False
    sat_z = False

    def get_xy_error(self, image, box):
        height, width = image.shape[0:2]
        center_x = (box[1] + box[3])*width/2
        center_y = (box[0] + box[2])*height/2

        err_x = (1 - 2*center_x/width)*width
        err_y = (1 - 2*center_y/height)*height

        return np.array([-err_x, err_y])

    def get_wh_error(self, image, box, percent=0.7):
        height, width = image.shape[0:2]

        err_w = (percent - (box[3] - box[1]))*width
        err_h = (percent - (box[2] - box[0]))*height

        return np.array([err_w, err_h])

    def pixel_to_angle(self, pixel_diff, fov, num_pixels):
        return fov*pixel_diff/num_pixels

    def angle_to_pixel(self, fov_diff, fov, num_pixels):
        return fov_diff*num_pixels/fov

    def fov_to_zoom(self, fov):
        # Curve coefficients
        p = [-10592.2948916415,
             604384.134517506,
             934552.144557281,
             -6170506.95490360,
             6786860.41011954]
        q = [-27.453859351470335,
             1.797183394439450e+04,
             -2.279948734246685e+04,
             -2.933962700625750e+04,
             6.383206210162090e+04]

        # Calculate polynomial power components
        fov_5 = fov**5.0
        fov_4 = fov**4.0
        fov_3 = fov**3.0
        fov_2 = fov**2.0

        # Convert fov to zoom
        zoom = ((p[0]*fov_4 + p[1]*fov_3 + p[2]*fov_2 + p[3]*fov + p[4])
                / (fov_5 + q[0]*fov_4 + q[1]*fov_3
                   + q[2]*fov_2 + q[3]*fov + q[4]))

        return zoom

    def pid_pt(self, image, boxes, feedback, tolerance=0.2):
        # PID parameters
        Kp = 0.01
        Kd = 0.001
        dt = 0.1

        if(boxes.size != 0):

            # Extract box
            box_i = boxes[0]

            # Compute error
            height, width = image.shape[0:2]
            err_xy = self.get_xy_error(image, box_i)
            err_pt = self.pixel_to_angle(err_xy, feedback["fov"], width)
            self.error.append(err_pt)

            # Generate PID control
            prop_pt = err_pt
            der_pt = np.array([0, 0])
            if(len(self.error) > 1):
                der_pt = (self.error[-1]-self.error[-2])/dt
            vel_pt = Kp*prop_pt + Kd*der_pt

            # Saturate
            if abs(err_pt[0]) < tolerance:
                vel_pt[0] = 0
            if abs(err_pt[1]) < tolerance:
                vel_pt[1] = 0

            if (all([v == 0 for v in vel_pt])):
                if(not self.sat_pt):
                    self.sat_pt = True
            else:
                self.sat_pt = False
        else:
            if(not self.sat_pt):
                self.sat_pt = True

        return (self.sat_pt, vel_pt)

    def pid_z(self, image, boxes, feedback, tolerance=0.1):
        # PID parameters
        Kp = 0.1
        Kd = 0.01
        dt = 0.1

        if(boxes.size != 0):

            # Extract box
            box_i = boxes[0]

            # Compute error
            height, width = image.shape[0:2]
            err_wh = self.get_wh_error(image, box_i)
            err = min(err_wh)
            err_fov = self.pixel_to_angle(-err, feedback["fov"], width)
            err_z = self.fov_to_zoom(err_fov)
            self.error_z.append(err_z)

            # Generate PID control
            prop = err_z
            der = 0
            if(len(self.error_z) > 1):
                der = (self.error_z[-1]-self.error_z[-2])/dt
            vel = Kp*prop + Kd*der

            if(abs(err_z) < tolerance):
                vel = 0

            if vel == 0:
                if(not self.sat_z):
                    self.sat_z = True
            else:
                self.sat_z = False
        else:
            if(not self.sat_z):
                self.sat_z = True

        return (self.sat_z, vel)


class PtzPidController:
    error = []
    error_z = []
    sat_pt = False
    sat_z = False

    def get_xy_error(self, image, box):
        height, width = image.shape[0:2]
        center_x = (box[1] + box[3])*width/2
        center_y = (box[0] + box[2])*height/2

        err_x = 1 - 2*center_x/width
        err_y = 1 - 2*center_y/height

        return np.array([-err_x, err_y])

    def get_wh_error(self, image, box, percent=0.7):
        height, width = image.shape[0:2]

        err_w = percent - (box[3] - box[1])
        err_h = percent - (box[2] - box[0])

        return np.array([err_w, err_h])

    def pid_pt(self, image, box, feedback=None, tolerance=0.05):
        """Run Pan-Tilt PID controller

        Parameters
        ----------
        image : ndarray of shape (h,w,3)
            The frame on which the box is laid out.
        box : ndarray of shape (1,4)
            The box coordinates given as [ymin, xmin, ymax, xmax]
        feedback : dict, optional
            Feedback received from the camera, by default None
        tolerance : float, optional
            Tolerance in Pan-Tilt expressed as fraction of the \
            image size, by default 0.2

        Returns
        -------
        dict
            A dict object containing control commands to the camera, e.g.
                * panSpeed
                * tiltSpeed
        dict
            A dict object containg the errors calculated by the pid:
                * prop: Proportional error
                * der: Derivative error
                * pid: Total error calculated the pid
        """
        # PID parameters
        Kp = 0.01
        Kd = 0.002

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1

        # Compute error
        err_xy = self.get_xy_error(image, box)
        self.error.append(err_xy)

        # Generate PID control
        prop_xy = err_xy
        der_xy = np.array([0, 0])
        if(len(self.error) > 1):
            der_xy = (self.error[-1]-self.error[-2])/dt
        vel_xy = np.round((Kp*prop_xy + Kd*der_xy)*255)/255.0

        # Saturate
        if abs(err_xy[0]) < tolerance:
            vel_xy[0] = 0
        if abs(err_xy[1]) < tolerance:
            vel_xy[1] = 0

        command = {
            "panSpeed": vel_xy[0],
            "tiltSpeed": vel_xy[1]
        }
        errors = {
            "prop": prop_xy,
            "der": der_xy,
            "pid": vel_xy
        }

        return command, errors

    def pid_z(self, image, box, feedback=None, tolerance=0.1):
        # PID parameters
        Kp = 100
        Kd = 10

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1

        # Compute error
        err_wh = self.get_wh_error(image, box)
        err = min(err_wh)
        self.error_z.append(err)

        # Generate PID control
        prop = err
        der = 0
        if(len(self.error_z) > 1):
            der = (self.error_z[-1]-self.error_z[-2])/dt
        vel = Kp*prop + Kd*der

        # print("ZoomSpeed: {}".format(vel))
        if(abs(err) < tolerance):
            vel = 0

        command = {
            "zoomRelative": vel*dt,
            "zoomSpeed": vel,
            "zoom": feedback["zoom"] + vel*dt
        }
        errors = {
            "prop": prop,
            "der": der,
            "pid": vel
        }

    #     if vel == 0:
    #         if(not self.sat_z):
    #             self.sat_z = True
    #             # Control the camera
    #             camera.zoom_http(vel)
    #     else:
    #         # Control the camera
    #         camera.zoom_http(vel)
    #         time.sleep(1*abs(vel))
    #         camera.zoom_http(0)
    #         self.sat_z = False
    # else:
    #     if(not self.sat_z):
    #         self.sat_z = True
    #         # Control the camera
    #         camera.zoom_http(0)

        return command, errors

    def run(self, image, box, feedback=None, tolerances=(0.2, 0.3)):
        pt_commands, pt_errors = self.pid_pt(
            image, box, feedback, tolerances[0])
        z_command, z_errors = self.pid_z(image, box, feedback, tolerances[1])

        command = {
            "PanTilt": pt_commands,
            "Zoom": z_command
        }

        errors = {
            "PanTilt": pt_errors,
            "Zoom": z_errors
        }

        return command, errors

class PtzPidController2:

    error_pan  = []
    error_tilt = []
    error_zoom = []
    i_terms = [0, 0, 0]
    sat_pt = False
    sat_z = False
    new_zoom = None

    def __init__(self, gains, tolerances=(0.2, 0.2, 0.3)):
        self.gains = gains
        self.tolerances = tolerances

    def reset(self):
        self.error_pan = []
        self.error_tilt = []
        self.error_zoom = []
        self.i_terms = [0, 0, 0]
        self.sat_pt = False
        self.sat_z = False

    def round(self, value):
        return np.round(value*255)/255.0

    def get_x_error(self, box):
        center_x = (box[1] + box[3])/2
        err_x = 0.5 - center_x
        return -err_x

    def get_y_error(self, box):
        center_y = (box[0] + box[2])/2
        err_y = 0.5 - center_y
        return err_y

    def get_wh_error(self, box):
        err_w = 0.9 - (box[3] - box[1])
        err_h = 0.9 - (box[2] - box[0])
        return np.array([err_w, err_h])

    def pid_pan(self, image, box, feedback=None, tolerance=None):
        """Run Pan-Tilt PID controller

        Parameters
        ----------
        image : ndarray of shape (h,w,3)
            The frame on which the box is laid out.
        box : ndarray of shape (1,4)
            The box coordinates given as [ymin, xmin, ymax, xmax]
        feedback : dict, optional
            Feedback received from the camera, by default None
        tolerance : float, optional
            Tolerance in Pan-Tilt expressed as fraction of the \
            image size, by default 0.2

        Returns
        -------
        dict
            A dict object containing control commands to the camera, e.g.
                * panSpeed
        dict
            A dict object containg the errors calculated by the pid:
                * prop: Proportional error
                * der: Derivative error
                * pid: Total error calculated the pid
        """
        # PID parameters
        Kp, Kd, Ki = self.gains[0]
        if tolerance is None:
            tolerance = self.tolerances[0]

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1

        # Compute error
        err = self.get_x_error(box)
        self.error_pan.append(err)
        self.i_terms[0] += err*dt

        # Generate PID control
        prop = err
        der = 0
        intg = self.i_terms[0]
        if(len(self.error_pan) > 1):
            der = (self.error_pan[-1]-self.error_pan[-2])/dt
        vel = self.round(Kp*prop + Ki*intg + Kd*der)

        # Saturate
        if abs(err) < tolerance:
            vel = 0
            self.i_terms[0] = 0

        command = {
            "panSpeed": vel
        }
        errors = {
            "prop": prop,
            "der": der,
            "pid": vel
        }

        return command, errors

    def pid_tilt(self, image, box, feedback=None, tolerance=None):
        """Run Pan-Tilt PID controller

        Parameters
        ----------
        image : ndarray of shape (h,w,3)
            The frame on which the box is laid out.
        box : ndarray of shape (1,4)
            The box coordinates given as [ymin, xmin, ymax, xmax]
        feedback : dict, optional
            Feedback received from the camera, by default None
        tolerance : float, optional
            Tolerance in Pan-Tilt expressed as fraction of the \
            image size, by default 0.2

        Returns
        -------
        dict
            A dict object containing control commands to the camera, e.g.
                * panSpeed
        dict
            A dict object containg the errors calculated by the pid:
                * prop: Proportional error
                * der: Derivative error
                * pid: Total error calculated the pid
        """
        # PID parameters
        Kp, Kd, Ki = self.gains[1]
        if tolerance is None:
            tolerance = self.tolerances[1]

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1

        # Compute error
        err = self.get_y_error(box)
        self.error_tilt.append(err)
        self.i_terms[1] += err*dt

        # Generate PID control
        prop = err
        der = 0
        intg = self.i_terms[0]
        if(len(self.error_tilt) > 1):
            der = (self.error_tilt[-1]-self.error_tilt[-2])/dt
        vel = self.round(Kp*prop + Ki*intg + Kd*der)

        # Saturate
        if abs(err) < tolerance:
            vel = 0
            self.i_terms[1] = 0

        command = {
            "tiltSpeed": vel
        }
        errors = {
            "prop": prop,
            "der": der,
            "pid": vel
        }

        return command, errors

    def pid_zoom(self, image, box, feedback=None, tolerance=None):
        # PID parameters
        Kp, Kd, Ki = self.gains[2]
        if tolerance is None:
            tolerance = self.tolerances[2]

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1
        feedback_zoom = round(feedback["zoom"])

        # Compute error
        err_wh = self.get_wh_error(box)
        err = min(err_wh)
        self.error_zoom.append(err)
        self.i_terms[2] += err*dt

        # Generate PID control
        prop = err
        der = 0
        intg = self.i_terms[2]
        if(len(self.error_zoom) > 1):
            der = (self.error_zoom[-1]-self.error_zoom[-2])/dt
        vel = Kp*prop + Ki*intg + Kd*der

        print("Feedback zoom: {}".format(feedback_zoom))
        print("New zoom: {}".format(self.new_zoom))
        if (self.new_zoom is not None and feedback_zoom != self.new_zoom) or (err > 0 and abs(err) < tolerance):
            vel = 0
            self.i_terms[2] = 0


        new_zoom = round(feedback_zoom + vel*dt)
        if self.new_zoom is None or (self.new_zoom is not None and self.new_zoom == feedback_zoom):
            self.new_zoom = new_zoom

        command = {
            "zoomRelative": vel*dt,
            "zoomSpeed": vel,
            "zoom": new_zoom
        }
        errors = {
            "prop": prop,
            "der": der,
            "pid": vel
        }
        return command, errors

    def run(self, image, box, feedback=None, tolerances=None):
        if tolerances is None:
            tolerances = self.tolerances

        p_commands, p_errors = self.pid_pan(
            image, box, feedback, tolerances[0])
        t_commands, t_errors = self.pid_tilt(
            image, box, feedback, tolerances[1])
        z_command, z_errors = self.pid_zoom(
            image, box, feedback, tolerances[2])

        command = {
            "Pan": p_commands,
            "Tilt": t_commands,
            "Zoom": z_command
        }
        errors = {
            "Pan": p_errors,
            "Tilt": t_errors,
            "Zoom": z_errors
        }

        return command, errors

class PtzPidController3:

    error_pan  = []
    error_tilt = []
    error_zoom = []
    i_terms = [0, 0, 0]
    sat_pt = False
    sat_z = False
    new_zoom = None
    last_command = {"type": "command",
                    "panSpeed": 0,
                    "tiltSpeed": 0,
                    "zoom": 0}
    last_command_time = datetime.now()

    def __init__(self, gains, tolerances=(0.2, 0.2, 0.3)):
        self.gains = gains
        self.tolerances = tolerances
        self.last_command_time = datetime.now()

    def reset(self):
        self.error_pan = []
        self.error_tilt = []
        self.error_zoom = []
        self.i_terms = [0, 0, 0]
        self.sat_pt = False
        self.sat_z = False

    def round(self, value):
        return np.round(value*255)/255.0

    def get_x_error(self, box):
        left = box[1]
        width = box[3]
        right = left + width
        center = (left + width)/2.0
        err_x = 0.5 - center
        # if (box[1] < 0.1):
        #     err_x1 = 0.1 - box[1]
        # if
        # err_x2 = 0.1 - box[1]
        # err_x3 = 0.9 - (box[1] + box[3])
        # return -min((err_x1, err_x2, err_x3))
        return -err_x

    def get_y_error(self, box):
        top = box[0]
        height = box[2]
        bottom = top + height
        center = (top + height)/2.0
        err_y = 0.5 - center
        return err_y

    def get_wh_error(self, box):
        err_w = 0.9 - (box[3] - box[1])
        err_h = 0.9 - (box[2] - box[0])
        return np.array([err_w, err_h])

    def pid_pan(self, image, box, feedback=None, tolerance=None):
        """Run Pan-Tilt PID controller

        Parameters
        ----------
        image : ndarray of shape (h,w,3)
            The frame on which the box is laid out.
        box : ndarray of shape (1,4)
            The box coordinates given as [ymin, xmin, ymax, xmax]
        feedback : dict, optional
            Feedback received from the camera, by default None
        tolerance : float, optional
            Tolerance in Pan-Tilt expressed as fraction of the \
            image size, by default 0.2

        Returns
        -------
        dict
            A dict object containing control commands to the camera, e.g.
                * panSpeed
        dict
            A dict object containg the errors calculated by the pid:
                * prop: Proportional error
                * der: Derivative error
                * pid: Total error calculated the pid
        """
        # PID parameters
        Kp, Kd, Ki = map(lambda x: float(x/(feedback["zoom"]/20.0)), self.gains[0])
        if tolerance is None:
            tolerance = self.tolerances[0]

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1

        # Compute error
        err = self.get_x_error(box)
        self.error_pan.append(err)
        self.i_terms[0] += err*dt

        # Generate PID control
        prop = err
        der = 0
        intg = self.i_terms[0]
        if(len(self.error_pan) > 1):
            der = (self.error_pan[-1]-self.error_pan[-2])/dt
        vel = self.round(Kp*prop + Ki*intg + Kd*der)

        # Saturate
        if abs(err) < tolerance:
            vel = 0
            self.i_terms[0] = 0

        command = {
            "panSpeed": vel
        }
        errors = {
            "prop": prop,
            "der": der,
            "pid": vel
        }

        return command, errors

    def pid_tilt(self, image, box, feedback=None, tolerance=None):
        """Run Pan-Tilt PID controller

        Parameters
        ----------
        image : ndarray of shape (h,w,3)
            The frame on which the box is laid out.
        box : ndarray of shape (1,4)
            The box coordinates given as [ymin, xmin, ymax, xmax]
        feedback : dict, optional
            Feedback received from the camera, by default None
        tolerance : float, optional
            Tolerance in Pan-Tilt expressed as fraction of the \
            image size, by default 0.2

        Returns
        -------
        dict
            A dict object containing control commands to the camera, e.g.
                * panSpeed
        dict
            A dict object containg the errors calculated by the pid:
                * prop: Proportional error
                * der: Derivative error
                * pid: Total error calculated the pid
        """
        # PID parameters
        Kp, Kd, Ki = map(lambda x: float(x/(feedback["zoom"]/20.0)), self.gains[1])
        if tolerance is None:
            tolerance = self.tolerances[1]

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1

        # Compute error
        err = self.get_y_error(box)
        self.error_tilt.append(err)
        self.i_terms[1] += err*dt

        # Generate PID control
        prop = err
        der = 0
        intg = self.i_terms[0]
        if(len(self.error_tilt) > 1):
            der = (self.error_tilt[-1]-self.error_tilt[-2])/dt
        vel = self.round(Kp*prop + Ki*intg + Kd*der)

        # Saturate
        if abs(err) < tolerance:
            vel = 0
            self.i_terms[1] = 0

        command = {
            "tiltSpeed": vel
        }
        errors = {
            "prop": prop,
            "der": der,
            "pid": vel
        }

        return command, errors

    def pid_zoom(self, image, box, feedback=None, tolerance=None):
        # PID parameters
        Kp, Kd, Ki = self.gains[2]
        if tolerance is None:
            tolerance = self.tolerances[2]

        if(feedback is not None and 'dt' in feedback):
            dt = feedback['dt']
        else:
            dt = 1
        feedback_zoom = round(feedback["zoom"])

        # Compute error
        err_wh = self.get_wh_error(box)
        err = min(err_wh)
        self.error_zoom.append(err)
        self.i_terms[2] += err*dt

        # Generate PID control
        prop = err
        der = 0
        intg = self.i_terms[2]
        if(len(self.error_zoom) > 1):
            der = (self.error_zoom[-1]-self.error_zoom[-2])/dt
        vel = self.round((Kp*prop + Ki*intg + Kd*der)/100.0)*100.0

        print("Feedback zoom: {}".format(feedback_zoom))
        print("New zoom: {}".format(self.new_zoom))
        if (err > 0 and abs(err) < tolerance):
            vel = 0
            self.i_terms[2] = 0


        new_zoom = round(feedback_zoom + vel*dt)

        command = {
            "zoomRelative": vel*dt,
            "zoomSpeed": vel,
            "zoom": new_zoom
        }
        errors = {
            "prop": prop,
            "der": der,
            "pid": vel
        }
        return command, errors

    def run(self, image, box, feedback=None, tolerances=None):
        if tolerances is None:
            tolerances = self.tolerances

        p_commands, p_errors = self.pid_pan(
            image, box, feedback, tolerances[0])
        t_commands, t_errors = self.pid_tilt(
            image, box, feedback, tolerances[1])
        z_command, z_errors = self.pid_zoom(
            image, box, feedback, tolerances[2])

        commands = {
            "Pan": p_commands,
            "Tilt": t_commands,
            "Zoom": z_command
        }
        errors = {
            "Pan": p_errors,
            "Tilt": t_errors,
            "Zoom": z_errors
        }

        panSpeed = commands["Pan"]["panSpeed"]
        tiltSpeed = commands["Tilt"]["tiltSpeed"]
        zoomRelative = commands["Zoom"]["zoomRelative"]
        zoomSpeed = commands["Zoom"]["zoomSpeed"]
        zoom = commands["Zoom"]["zoom"]

        # Pre-process commands to check convergence
        # This ensures that Pan-Tilt are allowed to converge first
        # after which the zoom will be adjusted. This helps make the
        # transformations less non-linear.
        converged = False
        if panSpeed + tiltSpeed == 0:
            if zoomRelative == 0:
                converged = True
        # else:
        #     # Control ONLY the pan-tilt
        #     zoomSpeed = 0
        #     zoom = feedback["zoom"]

        # Send command to CCTV server
        command = {"type": "command",
                   "panSpeed": panSpeed,
                   "tiltSpeed": tiltSpeed,
                   "zoom": zoom,
                   "zoomSpeed": zoomSpeed}

        send_flag = False
        if datetime.now() - self.last_command_time > timedelta(seconds=0.5):
            if not self.last_command == command:
                print("Sending command: {}".format(json.dumps(command)))
                self.last_command = command
                send_flag = True
                # if self.new_zoom is None or (self.new_zoom is not None and self.new_zoom == feedback["zoom"]):
                #     self.new_zoom = zoom
                # else:
                #     command["zoomSpeed"] = 0
                #     command["zoom"] = feedback["zoom"]

            else:
                print("Duplicate command: {}".format(json.dumps(command)))
            self.last_command_time = datetime.now()

        return command, send_flag, converged

if __name__ == "__main__":
    flir_pid = FlirPidController()
    for i in range(0, 70):
        print("{} -> {}".format(i, flir_pid.fov_to_zoom(i)))


