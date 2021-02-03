
from manimlib.imports import *

class graphx(GraphScene):
    #this sets up the axis configurations
    CONFIG={
        "x_min":0,
        "x_max":4,
        "y_min":0,
        "y_max":4,
        "function_color" : RED ,
        "axes_color" : GREEN,
        "x_tick_frequency": 0.25,
        "y_tick_frequency": 0.5,
        "x_labeled_nums" :range(0,4,1),
        "y_labeled_nums" :range(0,4,1),
        "x_axis_label":"$x$",
        "y_axis_label":"$y$",
    }
    #we can get the slope by looking at the data
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
    #the main graphing function
    def showfunction(self):

        x = [1, 2, 3]
        y = [1.2, 1.9, 3.2]
        #set the animation for getting the axes
        self.setup_axes(animate=True)
        #get the slopeand intercept to fit the line
        slop,intr = self.get_slp_intr(x,y)
        #this is a function of the line itself
        def func(z):
            return slop*z+intr
        #now use the function to draw a line between x=0 to x=3
        graph=self.get_graph(func,x_min=0,x_max=3)

        # add the given points to the 2d plane canvas
        for i in range(0,3):
            self.add(SmallDot(self.coords_to_point(x[i],y[i])))
            
        #play the drawing of the graph to be drawn
        self.play(ShowCreation(graph))
        #wait for 2 seconds
        self.wait(2)
        


    #the code execution starts from this place
    def construct(self):
        self.showfunction()
