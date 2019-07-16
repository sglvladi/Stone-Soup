import numpy as np
import time


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

    def pid_pt(self, image, box, feedback=None, tolerance=0.2):
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
        Kp = 0.005
        Kd = 0.01

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
        vel_xy = Kp*prop_xy + Kd*der_xy

        # Saturate
        if abs(err_xy[0]) < tolerance:
            vel_xy[0] = 0
        if abs(err_xy[1]) < tolerance:
            vel_xy[1] = 0

        command = {
            "panSpeed": vel_xy[0],
            "tiltSpeed": 0 #vel_xy[1]
        }
        errors = {
            "prop": prop_xy,
            "der": der_xy,
            "pid": vel_xy
        }

        # if (all([v == 0 for v in vel_xy])):
        #     if(not self.sat_pt):
        #         self.sat_pt = True
        #         # Control the camera
        #         camera.move_http(vel_xy)
        #         # camera.move_(vel_xy)
        # else:
        #     # Control the camera
        #     camera.move_http(vel_xy)
        #     # camera.move_(vel_xy/2)
        #     self.sat_pt = False
        # else:
        #     if(not self.sat_pt):
        #         self.sat_pt = True
        #         # Control the camera
        #         camera.move_http((0, 0))
        #         # camera.move_((0, 0))

        return command, errors

    def pid_z(self, image, box, feedback=None, tolerance=0.1):
        # PID parameters
        Kp = 0.2
        Kd = 0.01

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
            "zoomSpeed": vel
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

    def run(self, image, box, feedback=None, tolerances=(0.2, 0.1)):
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


if __name__ == "__main__":
    flir_pid = FlirPidController()
    for i in range(0, 70):
        print("{} -> {}".format(i, flir_pid.fov_to_zoom(i)))
