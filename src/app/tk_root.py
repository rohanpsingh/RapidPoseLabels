import tkinter as tk

class TkRoot:
    def __init__(self, window_title, img_width, img_height):
        self.width = img_width
        self.height = img_height

        # set up gui object
        self.tkroot = tk.Tk()
        self.tkroot.title(window_title)
        self.tkroot.geometry('900x500')

        widget_wd = 25
        widget_ht = 3
        # Button definitions and placements
        self.load_btn = tk.Button(self.tkroot, text="Load New Image",
                                  height=widget_ht, width=widget_wd,
                                  state=tk.NORMAL,
                                  command=self.btn_func_load)
        self.next_btn = tk.Button(self.tkroot, text="Next KeyPt",
                                  height=widget_ht, width=widget_wd,
                                  state=tk.DISABLED,
                                  command=self.btn_func_next)
        self.skip_btn = tk.Button(self.tkroot, text="Skip KeyPt",
                                  height=widget_ht, width=widget_wd,
                                  state=tk.DISABLED,
                                  command=self.btn_func_skip)
        self.reset_btn = tk.Button(self.tkroot, text="Reset",
                                   width=widget_wd,
                                   state=tk.DISABLED,
                                   command=self.btn_func_reset)
        self.scene_btn = tk.Button(self.tkroot, text="Next Scene",
                                   width=widget_wd,
                                   state=tk.DISABLED,
                                   command=self.btn_func_scene)
        self.compute_btn = tk.Button(self.tkroot, text="Compute",
                                     width=widget_wd,
                                     state=tk.DISABLED,
                                     command=self.btn_func_compute)
        self.display_btn = tk.Button(self.tkroot, text="Visualize",
                                     width=widget_wd,
                                     state=tk.DISABLED,
                                     command=self.btn_func_display)
        self.quit_btn = tk.Button(self.tkroot, text="Quit",
                                  width=widget_wd,
                                  state=tk.NORMAL,
                                  command=self.btn_func_quit)
        self.load_btn.grid(column=1, row=0, padx=10)
        self.next_btn.grid(column=1, row=1, padx=10)
        self.skip_btn.grid(column=1, row=2, padx=10)
        self.reset_btn.grid(column=1, row=3, padx=10)
        self.scene_btn.grid(column=1, row=4, padx=10)
        self.compute_btn.grid(column=1, row=5, padx=10)
        self.display_btn.grid(column=1, row=6, padx=10)
        self.quit_btn.grid(column=1, row=7, padx=10)

        # message box
        self.msg_box = tk.Label(self.tkroot,
                                text="Please load an image",
                                height = 5, width=widget_wd,
                                bg='blue', fg='white')
        self.dat_box = tk.Label(self.tkroot,
                                text="Current keypoint list:\n{}".format([]),
                                height = 10, width=widget_wd,
                                bg='blue', fg='white')
        self.msg_box.grid(column=1, row=8, padx=10)
        self.dat_box.grid(column=1, row=9, rowspan=3, padx=10, pady=10)

        # Create a canvas that can fit the image
        self.canvas = tk.Canvas(self.tkroot, width = self.width, height = self.height)
        self.canvas.grid(column=0, row=0, rowspan=10, padx=10, pady=10)
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill='blue')
        self.display_image = []

    def tkroot_main_loop(self):
        self.tkroot.mainloop()

    def btn_func_load(self):
        pass

    def btn_func_next(self):
        pass

    def btn_func_skip(self):
        pass

    def btn_func_reset(self):
        pass

    def btn_func_scene(self):
        pass

    def btn_func_compute(self):
        pass

    def btn_func_display(self):
        pass

    def btn_func_quit(self):
        self.tkroot.destroy()
