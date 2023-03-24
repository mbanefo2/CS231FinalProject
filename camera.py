import numpy as np

class Camera(object):
    def __init__(self, cam_name):
        
        # Camera Parameters
        if cam_name == 'iphonex':
            fx = 3149.102
            fy = 3148.456
            cx = 2012.223
            cy = 1457.441
            raDx = 0.214
            raDy = -0.703
            taDx = 0
            taDy = 0
            skew = 0
            R = np.vstack([
                [0.9977447439, -0.02744349417, -0.06125586154],
                [0.01105126748, 0.967308141, -0.2533630398],
                [0.06620646065, 0.2521146864, 0.9654298988]
                ])
            T = np.array([-83.91262614, -67.72431866, 370.9560732])
            T = T.reshape((3,1))
            
        elif cam_name == 'iphone13promax':
            fx = 2951.22
            fy = 2989.857
            cx = 2007.837
            cy = 1388.993
            raDx = 0.267
            raDy = -0.887
            taDx = 0
            taDy = 0
            skew = 0
            R = np.vstack([
                [0.9229406007, -0.05524817202, -0.3809570672],
                [-0.1025391938, 0.918605371, -0.3816410435],
                [0.3710341781, 0.3912950444, 0.8421530899]
                ])
            T = np.array([-100.4858498, -44.85467922, 289.6105319])
            T = T.reshape((3,1))
            
        else:
            raise ValueError('Camera name must be either iphonex or iphone13promax')
        
        self.camera_matrix = np.vstack([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.distortion_coef = np.array([raDx, raDy, taDx, taDy])
        self.extrinsic_matrix = np.hstack((R, T))
        self.camera_position = np.matmul(-np.linalg.inv(R), T)
        # self.court_corners = court_corners
        
    def image_to_camera(self, image_point):
        x, y = image_point[0], image_point[1]
        x, y = float(x), float(y)
        homogen_xyz = np.array([[x], [y], [1]])
        ray = np.dot(np.linalg.inv(self.camera_matrix), homogen_xyz)
        print(ray)
        ray /= np.linalg.norm(ray)
        print(ray)
        
        X = ray[0]
        Y = ray[1]
        Z = self.camera_position[2]
        p_camera = np.array([X, Y, Z, 1])
        p_world = np.dot(np.linalg.inv(self.extrinsic_matrix), p_camera)
        print(p_world)
        # # Convert to camera coordinate system
        # camera_ray = np.matmul(self.extrinsic_matrix[:, :3], ray) + self.extrinsic_matrix[:, 3]
        return p_world
    
    def camera_to_image(self, camera_point):
        P = np.dot(self.camera_matrix, self.extrinsic_matrix)
        cam_point_homogen = np.array([camera_point[0], camera_point[1], camera_point[2], 1])
        cam_point_homogen = np.reshape(cam_point_homogen, (4,1))
        
        # Project onto image plane
        img_homogen = np.dot(P, cam_point_homogen)
        u = img_homogen[0, 0] / img_homogen[2, 0]
        v = img_homogen[1, 0] / img_homogen[2, 0]
        return [u, v]

        