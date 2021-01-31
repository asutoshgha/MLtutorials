
from manimlib.imports import *

class graphx(GraphScene):

    CONFIG={
        "x_min":0,
        "x_max":4,
        "y_min":0,
        "y_max":4,
        "function_color" : RED ,
        "axes_color" : GREEN,
        "x_labeled_nums" :range(0,4,1),
        "y_labeled_nums" :range(0,4,1),
        "x_axis_label":"$x$",
        "y_axis_label":"$y$",
    }

    def get_slp_intr(self,x, y):
        n = len(x)

        x_mean = sum(x)/n
        y_mean = sum(y)/n

        num = 0
        denom = 0
        for i in range(n):
            num += (x[i] - x_mean)*(y[i] - y_mean)
            denom += (x[i] - x_mean)**2

        slope = num/denom
        intercept = y_mean - slope*x_mean
        return slope, intercept

    def showfunction(self):

        x = [1, 2, 3]
        y = [1.2, 1.9, 3.2]

        self.setup_axes(animate=True)
        points=[]
        slop,intr = self.get_slp_intr(x,y)
        def func(z):
            return slop*z+intr
        # point = Dot(self.coords_to_point(1,))
        graph=self.get_graph(func,x_min=0,x_max=3)

        #self.play(Write(text))
        #self.wait(1)
        #self.play(FadeOut(text))
        #self.wait(2)
        for i in range(0,3):
            self.add(SmallDot(self.coords_to_point(x[i],y[i])))
        self.play(ShowCreation(graph))
        self.wait(2)
        #slop,intr = self.get_slp_intr(x,y)

        # def func(z):
        #     return slop*z+intr

        # graph=self.get_graph(func,x_min=0,x_max=3)




    def construct(self):
        self.showfunction()
