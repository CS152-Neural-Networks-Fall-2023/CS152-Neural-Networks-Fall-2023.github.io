# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#| echo: false
import warnings
warnings.filterwarnings("ignore")
from manim import *
import autograd.numpy as np


class LectureScene(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera.background_color = "#ffffff"
        self.template = TexTemplate()
        self.template.add_to_preamble(r"\usepackage{amsmath}")

class ThreeDLectureScene(ThreeDScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera.background_color = "#ffffff"
        self.template = TexTemplate()
        self.template.add_to_preamble(r"\usepackage{amsmath}")
    

class VectorScene(LectureScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ax = Axes(
            x_range=[-7.5, 7.5, 1],
            y_range=[-5, 5, 1],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
        
        #axes_labels.set_color(GREY)
        self.add(self.ax)

class PositiveVectorScene(LectureScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ax = Axes(
            x_range=[-2.5, 12.5, 1],
            y_range=[-1, 9, 1],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
                #axes_labels.set_color(GREY)
        self.add(self.ax)

class ComparisonVectorScene(LectureScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ax1 = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": GREY},
        )
        self.ax2 = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": GREY},
        )
        axgroup = Group(self.ax1, self.ax2)
        axgroup.arrange_in_grid(buf=2)
        
        #axes_labels.set_color(GREY)
        self.add(axgroup)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none BasicFunction

class BasicFunction(PositiveVectorScene):
    def construct(self):
        fx = self.ax.plot(lambda x: 2 * (x / 5) ** 3 - 5 * (x / 6) ** 2 + 4, color=RED)
        eq = MathTex(r'f(\mathbf{x})=2(\frac{x_1}{5})^3 -5 (\frac{x_1}{6})^2 + 4', color=BLACK, tex_template=self.template).to_edge(UP + 2 * UP)
        #eq.move_to(*self.ax.c2p([8, 7, 0]))
        labels = self.ax.get_axis_labels(x_label="x_1", y_label="y = f(\mathbf{x})")
        labels.set_color(GREY)
        labels.set_tex_template(self.template)
        self.add(fx, eq, labels)
        
        
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none LinearFunction

class LinearFunction(PositiveVectorScene):
    def construct(self):
        fx = self.ax.plot(lambda x: 0.5 * x + 1, color=RED)
        eq = MathTex(r'f(\mathbf{x})= \frac{1}{2} x_1 + 1', color=BLACK, tex_template=self.template).to_edge(UP + 2 * UP)
        #eq.move_to(*self.ax.c2p([8, 7, 0]))
        labels = self.ax.get_axis_labels(x_label="x_1", y_label="y = f(\mathbf{x})")
        labels.set_color(GREY)
        labels.set_tex_template(self.template)
        self.add(fx, eq, labels)
        
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none ThreeDSurfacePlot

class ThreeDSurfacePlot(ThreeDLectureScene):
    def construct(self):
        resolution_fa = 24
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
        

        def param_gauss(u, v):
            x = u
            y = v
            return np.array([x, y, -0.6 * x + -0.2 * y - 1.])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-3, +3],
            u_range=[-3, +3]
        )

        gauss_plane.scale(1, about_point=ORIGIN)
        gauss_plane.set_style(fill_opacity=1, stroke_color=BLACK)
        gauss_plane.set_fill_by_checkerboard(RED, GREY, opacity=0.5)
        axes = ThreeDAxes()
        axes.set_color(GREY)
        labels = axes.get_axis_labels(x_label="x_1", y_label="x_2", z_label="y = f(\mathbf{x})")
        labels.set_color(GREY)
        #labels[0].rotate(75 * DEGREES, RIGHT)
        #labels[0].rotate(-30 * DEGREES, IN)
        #
        #labels[1].rotate(-30 * DEGREES, IN)

        eq = MathTex(r'f(\mathbf{x})= \frac{-3}{5} x_1 - \frac{1}{5} x_2 - 1', color=BLACK, tex_template=self.template)
        eq.to_corner(UL)
        eq.scale(0.8)
        self.add_fixed_in_frame_mobjects(eq)
        self.add_fixed_orientation_mobjects(labels[0])
        labels[1].rotate(90 * DEGREES, IN)
        self.add_fixed_orientation_mobjects(labels[1])
        labels[2].rotate(90 * DEGREES, LEFT)
        self.add_fixed_orientation_mobjects(labels[2])
        self.add( gauss_plane, axes, )
        
#
#
#
#
#
def f(x):
    w = np.array([-0.6, -0.2])
    b = -1
    return np.dot(x, w) + b
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
def f(x):
    w = np.array([-0.6, -0.2, -1])
    x = np.pad(x, ((0,1),), constant_values=1)
    return np.dot(x, w)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none HP_MPG
import pandas as pd
mpg_data = pd.read_csv('data/auto-mpg.csv')
data = mpg_data.sort_values(by=['weight'])[::50]

class HP_MPG(Scene):
    def construct(self):
        self.camera.background_color = "#ffffff"
        self.ax = Axes(
            x_range=[1000, 5000, 500],
            y_range=[10, 40, 10],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
        self.add(self.ax)
        
        
        coords = [[c1, c2] for (c1, c2) in zip(data['weight'], data['mpg'])]
        dots = VGroup(*[Dot(color=BLACK).move_to(self.ax.c2p(coord[0],coord[1])) for coord in coords])
        labels = self.ax.get_axis_labels(x_label="Weight\ (lbs)", y_label="MPG")
        labels.set_color(GREY)
        self.add(dots,  labels)
        for coord, name in list(zip(coords, data['car name'])):
            if name in ['honda accord cvcc', 'dodge aspen']:
                d = Dot(color=RED).move_to(self.ax.c2p(coord[0], coord[1]))
                label = Text('  '.join([n.capitalize() for n in name.split()[:2]]), color=RED, font_size=24).next_to(d, UR)
                self.add(d, label)
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none HP_MPG
import pandas as pd
mpg_data = pd.read_csv('data/auto-mpg.csv')
data = mpg_data.sort_values(by=['weight'])[::50]

class HP_MPG(Scene):
    def construct(self):
        self.camera.background_color = "#ffffff"
        self.ax = Axes(
            x_range=[1000, 5000, 500],
            y_range=[10, 40, 10],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
        self.add(self.ax)
        
        
        coords = [[c1, c2] for (c1, c2) in zip(data['weight'], data['mpg'])]
        dots = VGroup(*[Dot(color=BLACK).move_to(self.ax.c2p(coord[0],coord[1])) for coord in coords])

        query = Line(self.ax.c2p(3100, 10), self.ax.c2p(3100, 40), color=GREEN)
        qdot = Dot(self.ax.c2p(3100, 40))
        label = Text('Weight = 3100 lbs\nMPG = ?', color=GREEN, font_size=24).next_to(qdot, DR)
        self.add(label)
        
        self.add(query)
        labels = self.ax.get_axis_labels(x_label="Weight\ (lbs)", y_label="MPG")
        labels.set_color(GREY)
        self.add(dots,  labels)
        for coord, name in list(zip(coords, data['car name'])):
            if name in ['honda accord cvcc', 'dodge aspen']:
                d = Dot(color=RED).move_to(self.ax.c2p(coord[0], coord[1]))
                label = Text('  '.join([n.capitalize() for n in name.split()[:2]]), color=RED, font_size=24).next_to(d, UR)
                self.add(d, label)
                
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none HP_MPG
import pandas as pd
mpg_data = pd.read_csv('data/auto-mpg.csv')
data = mpg_data.sort_values(by=['weight'])[::50]

class HP_MPG(Scene):
    def construct(self):
        self.camera.background_color = "#ffffff"
        self.ax = Axes(
            x_range=[1000, 5000, 500],
            y_range=[10, 40, 10],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
        self.add(self.ax)
        
        
        coords = [[c1, c2] for (c1, c2) in zip(data['weight'], data['mpg'])]
        all_dots = [Dot(color=BLACK).move_to(self.ax.c2p(coord[0],coord[1])) for coord in coords]
        dots = VGroup(*all_dots)
        labels = self.ax.get_axis_labels(x_label="Weight\ (lbs)", y_label="MPG")
        labels.set_color(GREY)
        for d1, d2 in zip(all_dots[:-1], all_dots[1:]):
            self.add(Line(d1.get_center(), d2.get_center(), color=GREY))
        self.add(dots,  labels)
        for coord, name in list(zip(coords, data['car name'])):
            if name in ['honda accord cvcc', 'dodge aspen']:
                d = Dot(color=RED).move_to(self.ax.c2p(coord[0], coord[1]))
                label = Text('  '.join([n.capitalize() for n in name.split()[:2]]), color=RED, font_size=24).next_to(d, UR)
                self.add(d, label)
#
#
#
#
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none HP_MPG
import pandas as pd
mpg_data = pd.read_csv('data/auto-mpg.csv')
data = mpg_data.sort_values(by=['weight'])

class HP_MPG(Scene):
    def construct(self):
        self.camera.background_color = "#ffffff"
        self.ax = Axes(
            x_range=[1000, 5500, 500],
            y_range=[0, 50, 10],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
        self.add(self.ax)
        
        
        coords = [[c1, c2] for (c1, c2) in zip(data['weight'], data['mpg'])]
        all_dots = [Dot(color=BLACK).move_to(self.ax.c2p(coord[0],coord[1])) for coord in coords]
        dots = VGroup(*all_dots)
        labels = self.ax.get_axis_labels(x_label="Weight\ (lbs)", y_label="MPG")
        labels.set_color(GREY)
        for d1, d2 in zip(all_dots[:-1], all_dots[1:]):
            self.add(Line(d1.get_center(), d2.get_center(), color=GREY))
        self.add(dots,  labels)
        names = ['honda accord cvcc', 'dodge aspen']
        for coord, name in list(zip(coords, data['car name'])):
            
            if name in names:
                names.remove(name)
                d = Dot(color=RED).move_to(self.ax.c2p(coord[0], coord[1]))
                label = Text('  '.join([n.capitalize() for n in name.split()[:2]]), color=RED, font_size=24).next_to(d, UR)
                self.add(d, label)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
class Regression:
    def __init__(self, weights):
        self.weights = weights
    
    def predict(self, x):
        return np.dot(x, self.weights)

model = Regression(np.array([1, 1, 1, 1, 1]))
model.predict(np.array([5, 2, 3, 3, 1]))
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none HP_MPG
import pandas as pd
from sklearn.linear_model import LinearRegression
mpg_data = pd.read_csv('data/auto-mpg.csv')
data = mpg_data.sort_values(by=['weight'])[::50]

x = np.array(data['weight'])[:, np.newaxis]
y = np.array(data['mpg'])
model = LinearRegression().fit(x, y)

other_lines = [(model.coef_.item() * 1.2, model.intercept_.item() * 0.95), (model.coef_.item() * 0.7, model.intercept_.item() * 0.9), (model.coef_.item() * 1.8, model.intercept_.item() * 1.4)]

class HP_MPG(Scene):
    def construct(self):
        self.camera.background_color = "#ffffff"
        self.ax = Axes(
            x_range=[1000, 5000, 500],
            y_range=[10, 40, 10],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
        self.add(self.ax)
        
        
        coords = [[c1, c2] for (c1, c2) in zip(data['weight'], data['mpg'])]
        dots = VGroup(*[Dot(color=BLACK).move_to(self.ax.c2p(coord[0],coord[1])) for coord in coords])
        labels = self.ax.get_axis_labels(x_label="Weight\ (lbs)", y_label="MPG")
        labels.set_color(GREY)
        self.add(dots,  labels)
        for coord, name in list(zip(coords, data['car name'])):
            if name in ['honda accord cvcc', 'dodge aspen']:
                d = Dot(color=RED).move_to(self.ax.c2p(coord[0], coord[1]))
                label = Text('  '.join([n.capitalize() for n in name.split()[:2]]), color=RED, font_size=24).next_to(d, UR)
                self.add(d, label)

        # Add Regression line
        plot = self.ax.plot(lambda x: model.predict(np.atleast_2d(x)).item(), x_range=[1300, 4300], color=BLACK)
        self.add(plot)

        # Add other candidate lines
        for param, color in zip(other_lines, [BLUE, ORANGE, GREEN, PURPLE]):
            w, b = param
            plot = self.ax.plot(lambda x: x * w + b, x_range=[1300, 4300], color=color)
            self.add(plot)
        

        eq = Tex(r'$f(x)=wx+b$', color=BLACK).to_corner(UR)
        self.add(eq)
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
%%manim -sqh -v CRITICAL --progress_bar none HP_MPG_LR
from sklearn.linear_model import LinearRegression
x = np.array(mpg_data['weight'])[:, np.newaxis]
y = np.array(mpg_data['mpg'])
model = LinearRegression().fit(x, y)
other_lines = [(model.coef_.item() * 1.2, model.intercept_.item() * 0.95), (model.coef_.item() * 0.7, model.intercept_.item() * 0.9), (model.coef_.item() * 1.8, model.intercept_.item() * 1.4)]

class HP_MPG_LR(Scene):
    def construct(self):
        self.camera.background_color = "#ffffff"
        self.ax = Axes(
            x_range=[1000, 5000, 500],
            y_range=[10, 40, 10],
            x_length=12,
            y_length=8,
            axis_config={"color": GREY},
        )
        labels = self.ax.get_axis_labels(x_label="Weight\ (lbs)", y_label="MPG")
        labels.set_color(GREY)
        self.add(self.ax, labels)
        

        # Add data
        coords = [[c1, c2] for (c1, c2) in zip(data['weight'], data['mpg'])]
        all_dots = [Dot(color=BLACK).move_to(self.ax.c2p(coord[0],coord[1])) for coord in coords]
        dots = VGroup(*all_dots)
        self.add(dots)
        
        # Add highlighted points
        for coord, name in list(zip(coords, data['car name'])):
            if name in ['honda accord cvcc', 'dodge aspen']:
                d = Dot(color=RED).move_to(self.ax.c2p(coord[0], coord[1]))
                label = Text('  '.join([n.capitalize() for n in name.split()[:2]]), color=RED, font_size=24).next_to(d, UR)
                self.add(d, label)

        # Add Regression line
        plot = self.ax.plot(lambda x: model.predict(np.atleast_2d(x)).item(), x_range=[1300, 4300], color=BLACK)
        self.add(plot)

        eq = Tex(r'$f(x)=wx+b$', color=BLACK).to_corner(UR)
        self.add(eq)

        # Add residuals
        brace_num = 2
        for i, coord in enumerate(coords):
            x, y = coord
            yline = model.predict(np.atleast_2d(x)).item()
            line = Line(self.ax.c2p(x, y), self.ax.c2p(x, yline), color=GREY)
            self.add(line)
            if i == brace_num:
                resid_brace = Brace(line, direction=LEFT, color=GREY)
                resid_tex = resid_brace.get_tex(r"e_i=y_i-f(x_i)")
                resid_tex.set_color(BLACK)
                resid_tex.scale(0.8)
                self.add(resid_brace, resid_tex)

        model.predict(np.atleast_2d(x)).item()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
