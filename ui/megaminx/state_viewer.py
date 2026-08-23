"""Megaminx-specific state viewer."""

import tkinter as Tk

from megaminx.cube import DISPLAY_FACE_NAMES


class MegaminxStateViewer(Tk.Canvas):
    def __init__(self,master,mini_mode = False):
        scale = 0.55 if mini_mode else 1.0
        self.coordinate_corner = self._scale_points([(0,-17),(16,-5),(10,14),(-10,14),(-16,-5)], scale)
        self.coordinate_edge = self._scale_points([(0,14),(-13,4.5),(-8,-12),(8,-12),(13,4.5)], scale)
        self.corner_r = max(2, round(6 * scale))
        self.edge_r = max(2, round(4 * scale))
        self.center_r = max(4, round(10 * scale))

        D = 125
        center_x = 70 * scale
        center_y = 100 * scale
        self.center = [(0,0),(0,42.5),(-40,12.5),(-25,-35),(25,-35),(40,12.5),(D,-42.5),(D+40,-12.5),(D+25,35),(D-25,35),(D-40,-12.5),(D,0)]
        self.center = [((x[0] * scale) + center_x,(x[1] * scale) + center_y) for x in self.center]
        self.zero_position = [0,0,1,2,3,4,0,1,2,3,4,0]
        self.PM = [1,-1,-1,-1,-1,-1,1,1,1,1,1,-1]
        self.color = {'U':"#FFFFFF",
                      'F':"#7F0000",
                      'L':"#005F2F",
                      'P':"#3F007F",
                      'Q':"#DFDF00",
                      'R':"#0000BF",
                      'B':"#FF7F00",
                      'X':"#007FFF",
                      'G':"#FFDF7F",
                      'H':"#FF007F",
                      'Y':"#7FFF00",
                      'D':"#000000",
                      '':"#7F7F7F"}

        self.bd_color = self.color
        canvas_width = int(525 * scale)
        canvas_height = int(400 * scale)
        border_width = 2 if mini_mode else 4
        self.label_font = ('Comic Sans MS', max(10, round(20 * scale)), 'bold')
        Tk.Canvas.__init__(self,master,relief = Tk.RAISED , bd = border_width,width = canvas_width,height = canvas_height ,bg = '#BFBFBF')
        colors = ['U','F','L','P','Q','R','B','X','G','H','Y','D']

        for i in range(12):
            c = colors[i]
            self.create_oval(2 * (self.center[i][0] - self.center_r),
                                 2 * (self.center[i][1] - self.center_r),
                                 2 * (self.center[i][0] + self.center_r),
                                 2 * (self.center[i][1] + self.center_r),
                    fill = self.color[c],outline = self.bd_color[c],tags = 'centers')
            if i in [1,2,3,4,5,11]:
                fill_color = "#FFFFFF"
            else:
                fill_color = "#000000"
            self.create_text(2*self.center[i][0] ,2*self.center[i][1],text = DISPLAY_FACE_NAMES[c],font = self.label_font,fill = fill_color,tags = 'texts')

        
        



        


    def _scale_points(self, points, scale):
        """Scale a list of 2D points for the mini viewer."""
        return [(x * scale, y * scale) for x, y in points]

    def set_color(self,S):
        self.delete('circles')

        for i in range(120):
            if i % 10 < 5:
                self.create_oval(2 * (self.center[i // 10][0] + self.PM[i // 10]*self.coordinate_corner[(i+self.zero_position[i // 10]) % 5][0] - self.corner_r),
                                 2 * (self.center[i // 10][1] + self.PM[i // 10]*self.coordinate_corner[(i+self.zero_position[i // 10]) % 5][1] - self.corner_r),
                                 2 * (self.center[i // 10][0] + self.PM[i // 10]*self.coordinate_corner[(i+self.zero_position[i // 10]) % 5][0] + self.corner_r),
                                 2 * (self.center[i // 10][1] + self.PM[i // 10]*self.coordinate_corner[(i+self.zero_position[i // 10]) % 5][1] + self.corner_r),
                    fill = self.color[S[i]],outline = self.bd_color[S[i]],tags = 'circles')
            else:
                self.create_oval(2 * (self.center[i // 10][0] + self.PM[i // 10]*self.coordinate_edge[(i+self.zero_position[i // 10]) % 5][0] - self.edge_r),
                                 2 * (self.center[i // 10][1] + self.PM[i // 10]*self.coordinate_edge[(i+self.zero_position[i // 10]) % 5][1] - self.edge_r),
                                 2 * (self.center[i // 10][0] + self.PM[i // 10]*self.coordinate_edge[(i+self.zero_position[i // 10]) % 5][0] + self.edge_r),
                                 2 * (self.center[i // 10][1] + self.PM[i // 10]*self.coordinate_edge[(i+self.zero_position[i // 10]) % 5][1] + self.edge_r),
                    fill = self.color[S[i]],outline = self.bd_color[S[i]],tags = 'circles')                

        
                                
                                                                                        
        


State_viewer = MegaminxStateViewer
