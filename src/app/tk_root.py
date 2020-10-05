import tkinter as tk

class TkRoot:
    def __init__(self, window_title, img_width, img_height):
        self.width = img_width
        self.height = img_height

        # set up gui object
        self.tkroot = tk.Tk()
        self.tkroot.title(window_title)
        self.tkroot.geometry('880x500')
        self.widget_wd = 25
        self.widget_ht = 2

        #image on canvas
        self.display_image = []

        #layout
        self.init_layout()

    def tkroot_main_loop(self):
        self.tkroot.mainloop()

    def btn_func_load(self):
        pass

    def btn_func_skip(self):
        pass

    def btn_func_reset(self):
        pass

    def btn_func_next_scene(self):
        pass

    def btn_func_prev_scene(self):
        pass

    def btn_func_compute(self):
        pass

    def btn_func_display(self):
        pass

    def btn_func_create(self):
        pass

    def btn_func_choose(self):
        pass

    def btn_func_grasping(self):
        pass

    def btn_func_quit(self):
        self.tkroot.destroy()

    def main_layout(self):
        #destroy previous buttons
        for child in self.tkroot.winfo_children():
            child.destroy()

        # Button definitions and placements
        self.load_btn = tk.Button(self.tkroot, text="Load New Image",
                                  height=self.widget_ht, width=self.widget_wd,
                                  state=tk.NORMAL,
                                  command=self.btn_func_load)
        self.skip_btn = tk.Button(self.tkroot, text="Skip KeyPt",
                                  height=self.widget_ht, width=self.widget_wd,
                                  state=tk.DISABLED,
                                  command=self.btn_func_skip)
        self.reset_btn = tk.Button(self.tkroot, text="Reset",
                                   height=self.widget_ht, width=self.widget_wd,
                                   state=tk.DISABLED,
                                   command=self.btn_func_reset)
        self.next_scene_btn = tk.Button(self.tkroot, text="Next Scene",
                                        height=self.widget_ht, width=10,
                                        state=tk.DISABLED,
                                        command=self.btn_func_next_scene)
        self.prev_scene_btn = tk.Button(self.tkroot, text="Prev Scene",
                                        height=self.widget_ht, width=10,
                                        state=tk.DISABLED,
                                        command=self.btn_func_prev_scene)
        self.compute_btn = tk.Button(self.tkroot, text="Compute",
                                     height=self.widget_ht, width=self.widget_wd,
                                     state=tk.DISABLED,
                                     command=self.btn_func_compute)
        self.display_btn = tk.Button(self.tkroot, text="Visualize",
                                     height=self.widget_ht, width=self.widget_wd,
                                     state=tk.DISABLED,
                                     command=self.btn_func_display)
        self.quit_btn = tk.Button(self.tkroot, text="Quit",
                                  height=self.widget_ht-1, width=self.widget_wd,
                                  state=tk.NORMAL,
                                  command=self.btn_func_quit)
        self.load_slider = tk.Scale(self.tkroot, from_=0, to=999,
                                    orient=tk.HORIZONTAL,
                                    showvalue=0, length=200,
                                    command=self.btn_func_load)
        self.load_btn.place(x=self.width+20, y=10)
        self.load_slider.place(x=self.width+20, y=50)
        self.skip_btn.place(x=self.width+20, y=75)
        self.prev_scene_btn.place(x=self.width+20, y=115)
        self.next_scene_btn.place(x=self.width+125, y=115)
        self.reset_btn.place(x=self.width+20, y=155)
        self.compute_btn.place(x=self.width+20, y=195)
        self.display_btn.place(x=self.width+20, y=235)
        self.quit_btn.place(x=self.width+20, y=460)

        # message box
        self.msg_box = tk.Label(self.tkroot,
                                text="Please load an image",
                                height = 5, width=self.widget_wd+3,
                                bg='blue', fg='white')
        self.dat_box = tk.Label(self.tkroot,
                                text="Current keypoint list:\n{}".format([]),
                                height = 10, width=self.widget_wd+3,
                                bg='blue', fg='white')
        self.msg_box.place(x=self.width+20, y=275)
        self.dat_box.place(x=self.width+20, y=340)

        # Create a canvas that can fit the image
        self.canvas = tk.Canvas(self.tkroot, width = self.width, height = self.height)
        self.canvas.place(x=10, y=10)
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill='blue')

    def init_layout(self):
        #button for building a new model
        self.create_btn = tk.Button(self.tkroot, text="Create a new model",
                                    height=self.widget_ht, width=self.widget_wd,
                                    state=tk.NORMAL,
                                    command=self.btn_func_create)
        #button for choosing an existing model file
        self.choose_btn = tk.Button(self.tkroot, text="Use existing model",
                                    height=self.widget_ht, width=self.widget_wd,
                                    state=tk.NORMAL,
                                    command=self.btn_func_choose)
        #button for defining a grasping point
        self.grasping_btn = tk.Button(self.tkroot, text="Define grasp point",
                                      height=self.widget_ht, width=self.widget_wd,
                                      state=tk.NORMAL,
                                      command=self.btn_func_grasping)
        self.create_btn.place(x=350, y=150, height=50, width=200)
        self.choose_btn.place(x=350, y=220, height=50, width=200)
        self.grasping_btn.place(x=350, y=290, height=50, width=200)
