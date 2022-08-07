import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
import scipy.special


class Shape:
    ### ************************************************
    ### Constructor
    def __init__(self,
                 name          ='shape',
                 control_pts   =None,
                 n_control_pts =None,
                 n_sampling_pts=None,
                 radius        =None,
                 edgy          =None):
        if (name           is None): name           = 'shape'
        if (control_pts    is None): control_pts    = np.array([])
        if (n_control_pts  is None): n_control_pts  = 0
        if (n_sampling_pts is None): n_sampling_pts = 0
        if (radius         is None): radius         = np.array([])
        if (edgy           is None): edgy           = np.array([])

        self.name            = name
        self.control_pts     = control_pts
        self.n_control_pts   = n_control_pts
        self.n_sampling_pts  = n_sampling_pts
        self.curve_pts       = np.array([])
        self.area            = 0.0
        self.index           = 0
        self.bezier_pts_sets = []

        if (len(radius) == n_control_pts): self.radius = radius
        if (len(radius) == 1):             self.radius = radius*np.ones([n_control_pts])

        if (len(edgy) == n_control_pts):   self.edgy = edgy
        if (len(edgy) == 1):               self.edgy = edgy*np.ones([n_control_pts])

        subname             = name.split('_')
        if (len(subname) == 2):  # name is of the form shape_?.xxx
            self.name       = subname[0]
            index           = subname[1].split('.')[0]
            self.index      = int(index)
        if (len(subname) >  2):  # name contains several '_'
            print('Please do not use several "_" char in shape name')
            quit()

        if (len(control_pts) > 0):
            self.control_pts   = control_pts
            self.n_control_pts = len(control_pts)

    ### ************************************************
    ### Reset object
    def reset(self):
        self.name           = 'shape'
        self.control_pts    = np.array([])
        self.n_control_pts  = 0
        self.n_sampling_pts = 0
        self.radius         = np.array([])
        self.edgy           = np.array([])
        self.curve_pts      = np.array([])
        self.area           = 0.0

    ### ************************************************
    ### Generate shape
    def generate(self, *args, **kwargs):
        # Handle optional argument
        centering = kwargs.get('centering', True)
        cylinder  = kwargs.get('cylinder',  False)
        magnify   = kwargs.get('magnify',   1.0)

        # Generate random control points if empty
        if (len(self.control_pts) == 0):
            if (cylinder):
                self.control_pts = generate_cylinder_pts(self.n_control_pts)
            else:
                self.control_pts = generate_random_pts(self.n_control_pts)

        # Magnify
        self.control_pts *= magnify

        # Center set of points
        if (centering):
            center            = np.mean(self.control_pts, axis=0)
            self.control_pts -= center

        # Sort points counter-clockwise
        #control_pts, radius, edgy = ccw_sort(self.control_pts,
        #                                     self.radius,
        #                                     self.edgy)
        control_pts = np.array(self.control_pts)
        radius = np.array(self.radius)
        edgy = np.array(self.edgy)

        #self.control_pts = control_pts
        #self.radius      = radius
        #self.edgy        = edgy

        # Create copy of control_pts for further modification
        augmented_control_pts = control_pts

        # Add first point as last point to close curve
        augmented_control_pts = np.append(augmented_control_pts,
                                          np.atleast_2d(augmented_control_pts[0,:]), axis=0)

        # Compute list of cartesian angles from one point to the next
        vector = np.diff(augmented_control_pts, axis=0)
        angles = np.arctan2(vector[:,1],vector[:,0])
        wrap   = lambda angle: (angle >= 0.0)*angle + (angle < 0.0)*(angle+2*np.pi)
        angles = wrap(angles)

        # Create a second list of angles shifted by one point
        # to compute an average of the two at each control point.
        # This helps smoothing the curve around control points
        angles1 = angles
        angles2 = np.roll(angles,1)

        angles  = edgy*angles1 + (1.0-edgy)*angles2 + (np.abs(angles2-angles1) > np.pi)*np.pi

        # Add first angle as last angle to close curve
        angles  = np.append(angles, [angles[0]])

        # Compute curve segments
        local_curves = []
        bezier_pts_sets = []
        for i in range(0,len(augmented_control_pts)-1):
            local_curve, bezier_pts = generate_bezier_curve(augmented_control_pts[i,:],
                                                augmented_control_pts[i+1,:],
                                                angles[i],
                                                angles[i+1],
                                                self.n_sampling_pts,
                                                radius[i])
            local_curves.append(local_curve)
            bezier_pts_sets.append(bezier_pts)

        curve          = np.concatenate([c for c in local_curves])
        x, y           = curve.T
        z              = np.zeros(x.size)
        self.curve_pts = np.column_stack((x,y,z))
        self.curve_pts = remove_duplicate_pts(self.curve_pts)
        self.bezier_pts_sets = bezier_pts_sets


### ************************************************
### Compute distance between two points
def compute_distance(p1, p2):

    return np.sqrt(np.sum((p2-p1)**2))

### ************************************************
### Generate cylinder points
def generate_cylinder_pts(n_pts):
    if (n_pts < 4):
        print('Not enough points to generate cylinder')
        exit()

    pts = np.zeros([n_pts, 2])
    ang = 2.0*math.pi/n_pts
    for i in range(0,n_pts):
        pts[i,:] = [math.cos(float(i)*ang),math.sin(float(i)*ang)]

    return pts

### ************************************************
### Generate n_pts random points in the unit square
def generate_random_pts(n_pts):

    return np.random.rand(n_pts,2)

### ************************************************
### Compute minimal distance between successive pts in array
def compute_min_distance(pts):
    dist_min = 1.0e20
    for i in range(len(pts)-1):
        p1       = pts[i  ,:]
        p2       = pts[i+1,:]
        dist     = compute_distance(p1,p2)
        dist_min = min(dist_min,dist)

    return dist_min


### ************************************************
### Remove duplicate points in input coordinates array
### WARNING : this routine is highly sub-optimal
def remove_duplicate_pts(pts):
    to_remove = []

    for i in range(len(pts)):
        for j in range(len(pts)):
            # Check that i and j are not identical
            if (i == j):
                continue

            # Check that i and j are not removed points
            if (i in to_remove) or (j in to_remove):
                continue

            # Compute distance between points
            pi = pts[i,:]
            pj = pts[j,:]
            dist = compute_distance(pi,pj)

            # Tag the point to be removed
            if (dist < 1.0e-8):
                to_remove.append(j)

    # Sort elements to remove in reverse order
    to_remove.sort(reverse=True)

    # Remove elements from pts
    for pt in to_remove:
        pts = np.delete(pts, pt, 0)

    return pts

### ************************************************
### Compute Bernstein polynomial value
def compute_bernstein(n,k,t):
    k_choose_n = scipy.special.binom(n,k)

    return k_choose_n * (t**k) * ((1.0-t)**(n-k))

### ************************************************
### Sample Bezier curves given set of control points
### and the number of sampling points
### Bezier curves are parameterized with t in [0,1]
### and are defined with n control points P_i :
### B(t) = sum_{i=0,n} B_i^n(t) * P_i
def sample_bezier_curve(control_pts, n_sampling_pts):
    n_control_pts = len(control_pts)
    t             = np.linspace(0, 1, n_sampling_pts)
    curve         = np.zeros((n_sampling_pts, 2))

    for i in range(n_control_pts):
        curve += np.outer(compute_bernstein(n_control_pts-1, i, t),
                          control_pts[i])

    return curve

### ************************************************
### Generate Bezier curve between two pts
def generate_bezier_curve(p1, p2, angle1, angle2, n_sampling_pts, radius):
    # Sample the curve if necessary
    if (n_sampling_pts != 0):
        dist = compute_distance(p1, p2)
        if (radius == 'random'):
            radius = 0.707*dist*np.random.uniform(low=0.0, high=1.0)
        else:
            radius = 0.707*dist*radius

            # Create array of control pts for cubic Bezier curve
            # First and last points are given, while the two intermediate
            # points are computed from edge points, angles and radius
            control_pts      = np.zeros((4,2))
            control_pts[0,:] = p1[:]
            control_pts[3,:] = p2[:]
            control_pts[1,:] = p1 + np.array(
                [radius*np.cos(angle1), radius*np.sin(angle1)])
            control_pts[2,:] = p2 + np.array(
                [radius*np.cos(angle2+np.pi), radius*np.sin(angle2+np.pi)])

            # Compute points on the Bezier curve
            curve = sample_bezier_curve(control_pts, n_sampling_pts)

    # Else return just a straight line
    else:
        curve = p1
        curve = np.vstack([curve,p2])

    return curve, control_pts

### ************************************************
### Plot the shape
def plot_shape(shape, **kwargs):
    r_max = kwargs.get('rmax', 5)
    r_min = kwargs.get('rmin', 1)
    x_min = -r_max - 2
    y_min = -r_max - 2
    x_max = r_max + 2
    y_max = r_max + 2

    n_pts = shape.n_control_pts

    fig = plt.figure(figsize=(7, 7))
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    # Line and Circle
    for n in np.arange(n_pts):
        line = plt.Line2D([0, (r_max+1)*np.cos(2*np.pi/n_pts*n)], [0, (r_max+1)*np.sin(2*np.pi/n_pts*n)],
                          color='brown', ls='--', lw=1)
        fig.gca().add_artist(line)
    circle1 = plt.Circle((0, 0), r_min, ls='--', color='brown', fill=False)
    circle2 = plt.Circle((0, 0), r_max, ls='--', color='brown', fill=False)
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)

    handle_control_pts = plt.scatter(shape.control_pts[:, 0], shape.control_pts[:, 1], label='Control points')
    plt.plot(np.append(shape.curve_pts[:, 0], shape.curve_pts[0, 0]),
             np.append(shape.curve_pts[:, 1], shape.curve_pts[0, 1]))
    for n in range(n_pts):
        plt.plot(shape.bezier_pts_sets[n][:, 0], shape.bezier_pts_sets[n][:, 1], ls='-.', lw=0.5, color='gray')
        handle_auxiliary_pts = plt.scatter(shape.bezier_pts_sets[n][1:3, 0], shape.bezier_pts_sets[n][1:3, 1],
                                           marker='.', s=20, color='gray', label='Augment points')

    plt.legend(handles=[handle_control_pts, handle_auxiliary_pts])
    plt.show()
    pass


### ************************************************
### Generate control points
def generate_control_points(n_pts, r_min, r_max, radius_portion=None, angle_portion=None):
    if radius_portion is None:
        radius_portion = np.random.rand(n_pts)
    if angle_portion is None:
        angle_portion = np.random.rand(n_pts)
    radius = r_min + (r_max - r_min) * radius_portion
    angle = np.linspace(0, 2*np.pi, n_pts+1)[0:-1] + 2*np.pi/n_pts * angle_portion
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    deg = angle*180/np.pi
    print(deg)
    control_pts = np.column_stack((x, y))
    print(control_pts)
    return control_pts


if __name__ == '__main__':
    n_pts = 6
    r_min = 1
    r_max = 5
    control_pts = generate_control_points(n_pts=n_pts, r_min=r_min, r_max=r_max)
    shape = Shape(name          ='shape',
                  control_pts   =control_pts,
                  n_control_pts =n_pts,
                  n_sampling_pts=20,
                  radius        =0.5*np.ones([n_pts]),
                  edgy          =0.5*np.ones([n_pts]))
    shape.generate(centering=False)
    # print(shape.curve_pts)
    plot_shape(shape, r_min=r_min, r_max=r_max)
    pass
