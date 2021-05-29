import tkinter.filedialog
import tkinter as tk
import cv2, time
import numpy as np
from PIL import Image, ImageTk
from threading import Thread
from BackgroundRemover import U2NETPredictor

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.predictor = U2NETPredictor()
        self.pack()
        
        self.frame = np.zeros((500,500), dtype="uint8")

        self.bg_loaded = False
        self.cam_state = False
        cam_thread = Thread(target=self.startCam)
        cam_thread.daemon = True
        cam_thread.start()
        shw_image = Thread(target=self.show_img)
        shw_image.daemon = True
        shw_image.start()

        self.create_widgets()

    def open_backgroud(self):
        if self.cam_state:
            file = tkinter.filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            h,w = self.frame.shape[:2]
            self.bg = cv2.resize(cv2.imread(file), (w,h))
            self.bg_loaded = True

    def startCam(self):
        while True:
            if self.cam_state:
                cam = cv2.VideoCapture(0)

                if cam.isOpened():
                    rval, frame = cam.read()  
                    i = 0
                    while rval:
                        rval, frame = cam.read()
                        if not self.cam_state:
                            self.frame = np.zeros((500,500), dtype="uint8")
                            cam.release()
                            break
                        elif self.bg_loaded:
                            if i%3 == 0:
                                pred = self.predictor.predict(frame)
                            coords = np.where(pred < 50)
                            frame[coords] = self.bg[coords]
                            i+=1
                        self.frame = frame
            else:
                time.sleep(0.1)
            
    def change_cam_state(self):
        self.cam_state = not self.cam_state
        if self.bg_loaded: self.bg_loaded = False
        self.btn_start["text"] = "Stop" if self.btn_start["text"]=="Start" else "Start"

    def show_img(self):
        while True:
            frame = cv2.cvtColor(cv2.resize(self.frame, (500,500)), cv2.COLOR_BGR2RGB)
            frame =  ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.my_img.imgtk = frame
            self.my_img.configure(image=frame)
            if not self.cam_state: time.sleep(0.5)
            else: time.sleep(0.033)

    def create_widgets(self):
        self.btn_start = tk.Button(self.master, text="Start", font='sans 12 bold', fg='white', bg='DeepSkyBlue2', command = self.change_cam_state)
        self.btn_start.place(relx=0.2, rely=0.06, height = 50, width = 100, anchor="center")

        self.btn_open = tk.Button(self.master, text="Open Background", font='sans 12 bold', fg='white', bg='slateGray3', command = self.open_backgroud)
        self.btn_open.place(relx=0.48, rely=0.06, height = 50, width = 150, anchor="center")
        

        #self.show_img(self.img)
        self.my_img = tk.Label(self.master)
        self.my_img.place(relx=0.5, rely=0.55, height = 500, width = 500, anchor="center")
        #self.show_img()

        self.quit = tk.Button(self.master, text="QUIT", font='sans 12 bold',fg="white", bg="firebrick3", command=self.master.destroy)
        self.quit.place(relx=0.78, rely=0.06, height = 50, width = 100, anchor="center")



root = tk.Tk()
width, height = 600,600
root.geometry(f"{width}x{height}");root.minsize(width, height);root.maxsize(width, height)
app = Application(master=root)
app.mainloop()
