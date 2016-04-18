#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
import Tkinter
import ttk
import tkFileDialog
import dbcolonysizer

class simpleapp_tk(Tkinter.Tk):
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()

        s=ttk.Style()
        s.theme_use('clam')

# directory button
        self.imagedirectory = Tkinter.StringVar()
        self.directory_textentry = ttk.Entry(self,textvariable=self.imagedirectory)
        self.directory_textentry.grid(column=0,row=0,sticky='EW')
        self.imagedirectory.set(u"...")

        directory_button = ttk.Button(self,text=u"Choose image directory",
                                command=self.OnDirButtonClick)
        directory_button.grid(column=1,row=0)

# kernel button
        self.kernel_path = Tkinter.StringVar()
        self.kernel_path_textentry = ttk.Entry(self,textvariable=self.kernel_path)
        self.kernel_path_textentry.grid(column=0,row=1,sticky='EW')
        self.kernel_path.set(u"...")

        kernel_button = ttk.Button(self,text=u"Choose example crop image",
                                command=self.OnKernelButtonClick)
        kernel_button.grid(column=1,row=1)

# Go button
        go_button = ttk.Button(self,text=u"Start",
                                command=self.OnGoButtonClick)
        go_button.grid(column=1,row=2, columnspan=2)

        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,False)
        self.directory_textentry.focus_set()
        self.directory_textentry.selection_range(0, Tkinter.END)

    def OnDirButtonClick(self):
        self.imagedirectory.set(tkFileDialog.askdirectory(**{}) )
        self.directory_textentry.focus_set()
        self.directory_textentry.selection_range(0, Tkinter.END)

    def OnKernelButtonClick(self):
        self.kernel_path.set(tkFileDialog.askopenfilename() )
        self.kernel_path_textentry.focus_set()
        self.kernel_path_textentry.selection_range(0, Tkinter.END)

    def OnGoButtonClick(self):
        print self.imagedirectory.get()
        print self.kernel_path.get()
        files = dbcolonysizer.get_file_list(image_directory = self.imagedirectory.get(), template_image = self.kernel_path.get())
        dbcolonysizer.process_files(files)
        self.directory_textentry.focus_set()
        self.directory_textentry.selection_range(0, Tkinter.END)

if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('DB Colony Sizer')
    app.mainloop()
