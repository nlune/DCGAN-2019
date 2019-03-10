#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Frame, Button, Style, Progressbar
from tkinter import filedialog
from PIL import Image, ImageTk
import preinpaint
import inpaint
import postinpaint
from scipy.misc import toimage
from skimage.io import imsave
import imageio
import config
import functions as func
import datetime

class InpaintApp(Frame):

    def __init__(self):
        super().__init__()

        self.initInpaintUI()


    def initInpaintUI(self):
        self.master.title("Fill Your Face")

        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(4, pad=2)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, pad=2)

        lbl = Label(self, text="Please select a mask type & face to fill.")
        lbl.grid(sticky=W, pady=4, padx=5)

        wbtn = Button(self, text="Choose Folder", command=self.saveLoc)
        wbtn.place(x=10, y=30)


        # variables
        self.save_loc = None
        self.chosen_img = None
        self.completed_img = None
        self.msk = None

        self.lr = config.learning_rate
        self.moment = config.momentum
        self.nitrs = config.niters
        self.l2loss = False
        self.saveEaItr = False # saves images after ea iteration to visualize manifold
        self.weighted_mask = config.weighted_mask

        self.show_mskedim = False

        self.saveMask = IntVar() # checkboxes whether to save as image
        self.saveMaskedIm = IntVar()

        # buttons right
        abtn = Button(self, text="Select Image", command=self.openImg)
        abtn.grid(row=2, column=4, padx =5, pady=5)




        abtn = Button(self, text="Random Image", command=self.getRandImg)
        abtn.grid(row=1, column=4, padx =5, pady=5)

        # choose mask
        choices = {'Center','Random', 'Half' }
        self.mask_type = StringVar(root)
        self.mask_type.set("Select Mask Type")
        popupMenu = OptionMenu(self, self.mask_type, *choices, command=self.display_mask)
        popupMenu.grid(row = 1, column =3, padx=5)



        dmbtn = Button(self, text="Display Masked Image", command=self.display_masked_img)
        dmbtn.grid(row=2, column=3, padx=5)


        # buttons lower
        hbtn = Button(self, text="Help", command=self.clickHelp)
        hbtn.grid(row=5, column=0, padx=5, pady=5)

        wbtn = Button(self, text="Settings", command=self.setParams)
        wbtn.grid(row=5, column=1)

        Button(self, text="Save Image(s)", command=self.saveImg).grid(row=5, column=2, padx=5, pady=5)


        obtn = Button(self, text="Start Inpaint", command=self.start_inpaint)
        obtn.grid(row=5, column=3, padx=5, pady=5)


        cbtn = Button(self, text="Quit", command=self.ExitApplication)
        cbtn.grid(row=5, column=4, padx=5, pady=5)


        # progress bar
        Label(self, text="Inpainting Progress:").place(x=40, y=260)
        self.progressbar = Progressbar(self, length=500)
        # self.progressbar.config(mode = 'determinate',  maximum=self.nitrs)
        self.progressbar.place(x = 40, y = 280)


    def ExitApplication(self):
        MsgBox = messagebox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
        if MsgBox == 'yes':
           root.destroy()

    def openImg(self):
        filename = filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
        if filename[-4:] == '.png' or filename[-4:] == 'jpeg' or filename[-4:] == '.jpg':
            img = Image.open(filename)
            self.blend_tar = np.asarray(img) # for poisson blending later
            self.blend_tar.flags.writeable = True
            if img.size != (64,64):
                messagebox.showwarning("Error", "Please choose a 64x64 rgb face image.\
                \n You can use openface to crop your image. Click Help for more info.")
                return
            self.chosen_img = np.asarray(img, dtype="float32" )
            img = img.resize((100, 100), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(img)
            txt_lbl = Label(self, text="Your Image")
            txt_lbl.place(x=50, y=110)
            dis_img = Label(self, image=imgtk)
            dis_img.image = imgtk
            dis_img.place(x=50, y=127)
        else:
            messagebox.showwarning("Error", "Please select a png or jpeg image")

    def getRandImg(self):
        dataset = tf.data.TFRecordDataset(filenames="data.tfrecord")
        dataset  = dataset.map(func.extract_fn)

        # dataset = tf.data.TFRecordDataset.from_tensor_slices(dataset)
        dataset = dataset.shuffle(buffer_size=300)
        dataset = dataset.batch(1)


        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        init_op = iterator.make_initializer(dataset)

        # batched data to feed in
        image_data = iterator.get_next()

        # visualize original data
        with tf.Session() as sess:
            sess.run(init_op)
            real_img = np.array(sess.run(image_data))
            rand_img = real_img[0,:,:,:]
            self.chosen_img = np.asarray(rand_img, dtype="float32")
            img = toimage(rand_img)
            self.blend_tar = np.asarray(img) # for poisson blending later
            self.blend_tar.flags.writeable = True
            img = img.resize((100, 100), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(img)
            txt_lbl = Label(self, text="Your Image")
            txt_lbl.place(x=50, y=110)
            dis_img = Label(self, image=imgtk)
            dis_img.image = imgtk
            dis_img.place(x=50, y=127)



    def saveLoc(self):
        self.save_loc = filedialog.askdirectory()
        self.dis_save = Label(self, text = 'Save Location: ')
        self.dis_save.place(x=10, y=60)
        slength = len(self.save_loc)

        if slength > 40:
            self.dis_save['text'] += "..."
            self.dis_save['text'] += self.save_loc[-40:]
        else:
            self.dis_save['text'] += self.save_loc

    def saveImg(self):
        if self.completed_img is not None:
            im = toimage(self.completed_img)
            im = im.resize((250, 250), Image.ANTIALIAS)

            if self.save_loc is not None:
                # file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
                uniq_filename = 'CompletedImg_' + str(datetime.datetime.now().time()).replace(':', '.')
                filename = self.save_loc + '/' + uniq_filename + '.jpg'
                imageio.imwrite(filename, im)
                if self.saveMask.get():
                    uniq_filename = 'Mask_' + str(datetime.datetime.now().time()).replace(':', '.')
                    filename = self.save_loc + '/' + uniq_filename + '.jpg'
                    mskimg = toimage(self.msk)
                    mskimg = mskimg.resize((250, 250), Image.ANTIALIAS)
                    imageio.imwrite(filename, mskimg)

                if self.saveMaskedIm.get():
                    uniq_filename = 'MaskedImg_' + str(datetime.datetime.now().time()).replace(':', '.')
                    filename = self.save_loc + '/' + uniq_filename + '.jpg'
                    masked_img = preinpaint.get_masked_image(self.msk, self.chosen_img)
                    masked_img = toimage(masked_img)
                    masked_img = masked_img.resize((250, 250), Image.ANTIALIAS)
                    imageio.imwrite(filename, masked_img)

                messagebox.showinfo("Success!", "Your image has been saved")
            else: messagebox.showwarning("Error", "Please choose folder to save image.")

        else:
            messagebox.showwarning("Error", "Please inpaint first to get image.")


    def display_mask(self, mask_type):
        if mask_type != "Select Mask Type":
            self.msk = preinpaint.make_mask(mask_type, weighted_mask=self.weighted_mask)
            # mskimg = Image.fromarray(msk, 'RGB')
            # dispmsk = preinpaint.make_mask(mask_type, weighted_mask=False)
            mskimg = toimage(self.msk)
            mskimg = mskimg.resize((100, 100), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(mskimg)
            txt_lbl = Label(self, text="Your Mask: ")
            txt_lbl.place(x=200, y=110)
            dis_img = Label(self, image=imgtk)
            dis_img.image = imgtk
            dis_img.place(x=200, y=127)
        else:
            messagebox.showwarning("Error", "Please select a mask first!")


    def display_masked_img(self):
        if self.chosen_img is not None and self.msk is not None:
            masked_img = preinpaint.get_masked_image(self.msk, self.chosen_img)
            masked_img = toimage(masked_img)
            masked_img = masked_img.resize((100, 100), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(masked_img)
            txt_lbl = Label(self, text="Masked Image: ")
            txt_lbl.place(x=350, y=110)
            dis_img = Label(self, image=imgtk)
            dis_img.image = imgtk
            dis_img.place(x=350, y=127)
            self.show_mskedim = True
        else:
            messagebox.showwarning("Error", "Please load image and mask")


    def start_inpaint(self):
        l2 = self.l2loss
        savegenerated = self.saveEaItr

        if self.chosen_img is not None and self.msk is not None:
            self.progressbar['value'] = 0
            self.progressbar.update_idletasks()

            img = preinpaint.preprocess(self.chosen_img)
            # print(img.dtype, np.amax(img), np.amin(img)) # debugging
            images = preinpaint.single_to_batch(img)
            masks = preinpaint.single_to_batch(self.msk)

            # gen_images, loss = inpaint.get_best_z_img(masks, images, iters=self.nitrs)
            # backprop to z
            z = np.random.randn(config.BATCH_SIZE, config.z_dim)
            vel = 0
            iters = self.nitrs
            r = self.lr
            momentum = self.moment

            self.progressbar.config(mode = 'determinate',  maximum=self.nitrs)

            # load frozen graph
            graph, graph_def = func.loadpb("dcgan-100.pb")

            g_in = graph.get_tensor_by_name('dcgan/z:0')
            g_out = graph.get_tensor_by_name('dcgan/Tanh:0')
            g_loss = graph.get_tensor_by_name('dcgan/Mean_2:0')
            d_in = graph.get_tensor_by_name('dcgan/real_images:0')
            d_out = graph.get_tensor_by_name('dcgan/Sigmoid:0')


            with tf.Session(graph=graph) as sess:
                # create batches of masks & images to feed in
                mask_placeholder = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE,config.image_size, config.image_size, config.channels))
                # real image batches, use if own image input
                images_placeholder = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE,config.image_size, config.image_size, config.channels))

                inpaint_loss, gradient = inpaint.get_losses(mask_placeholder, images_placeholder, g_in,g_out,g_loss, l2)
                bests = []
                for i in range(iters):
                    # yield Label(self, text="Inpainting Progress:" + str(i)).place(x=40, y=280)
                    feed_dict = {mask_placeholder: masks, images_placeholder: images, g_in: z}
                    loss, grad, gen_images = sess.run((inpaint_loss, gradient, g_out), feed_dict=feed_dict)

                    grad = grad[0]


                    v_prev = vel
                    vel = v_prev*momentum - r*grad
                    # z += vel
                    z += vel*(1 + momentum) - v_prev*momentum   # dampening momentum

                    z = np.clip(z, -1, 1)

                    # debugging -- save best gen img of ea. iteration
                    if savegenerated:
                        best = inpaint.get_best_generated(gen_images, loss)
                        bests.append(best)
                        for b in bests:
                            im = toimage(b)
                            im.save("generated/z_" + str(i) + ".jpg")

                    self.progressbar['value'] = i+1
                    self.progressbar.update_idletasks()

            best_image = inpaint.get_best_generated(gen_images, loss)

            # poisson blending
            blend_src = np.asarray(toimage(best_image))

            mask = preinpaint.bin_inv_mask(self.msk)
            mask = np.asarray(toimage(mask))

            self.completed_img = postinpaint.blend(self.blend_tar, blend_src, mask)
            # print(self.blend_tar.shape, blend_src.shape, mask.shape)
            # self.completed_img = inpaint.inpaint(img, best_image, self.msk)
            disp_img = toimage(self.completed_img)
            disp_img = disp_img.resize((128, 128), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(disp_img)
            txt_lbl = Label(self, text="Completed Image: ")
            txt_lbl.place(x=250, y=330)
            dis_img = Label(self, image=imgtk)
            dis_img.image = imgtk
            dis_img.place(x=250, y=347)



    def clickHelp(self):
        toplevel = Toplevel(root)
        toplevel.title("Help")
        with open("files/about.txt", "r") as f:
            Label(toplevel, text=f.read(), height=20, width=100).pack()

        def closetl():
            toplevel.destroy()

        Button(toplevel, text="Close", command=closetl).pack()


    def setParams(self):
        tl = Toplevel(root)
        tl.title("Set Parameters")
        Label(tl, text="Iterations: ").grid(row=0, column=0, sticky=W, padx = 5, pady=3)
        self.iters = Entry(tl, width=10)
        self.iters.grid(row=0, column=1, padx=5, pady=3)
        Label(tl, text="Use weighted mask?: ").grid(row=1, column=0,padx=5, pady=3)
        self.maskedbool = BooleanVar()
        Radiobutton(tl, text="Yes", variable= self.maskedbool, value=True).grid(row=1, column=1)
        Radiobutton(tl, text="No", variable= self.maskedbool, value=False).grid(row=1, column=2)
        # Checkbutton(tl, text="Use weighted mask?", variable=self.maskedbool).grid(row=1, column=3)
        ch1=Checkbutton(tl, text="Save mask?", variable=self.saveMask)
        ch1.grid(row=2, column=0,padx=5, pady=7)
        ch2=Checkbutton(tl, text="Save masked image?", variable=self.saveMaskedIm)
        ch2.grid(row=2, column=1,padx=5, pady=7)

        def closetl():
            tl.destroy()

        def changeParams():
            try:
                if int(self.iters.get()) > 0:
                    self.nitrs = int(self.iters.get())
                else: messagebox.showwarning("Error", "You need at least 1 iteration.")
                self.weighted_mask = self.maskedbool.get()
                if self.mask_type.get() != "Select Mask Type":
                    self.display_mask(self.mask_type.get())
                    if self.show_mskedim:
                        self.display_masked_img()

                tl.destroy()
            except ValueError:
                self.weighted_mask = self.maskedbool.get()
                if self.mask_type.get() != "Select Mask Type":
                    self.display_mask(self.mask_type.get())
                tl.destroy()

        Button(tl, text="Ok", command=changeParams).grid(row=3, column=2, padx=5, pady=3)
        Button(tl, text="Cancel", command=closetl).grid(row=3, column=3, padx=5, pady=3)





if __name__ == '__main__':
        root = Tk()
        root.resizable(0, 0)
        app = InpaintApp()
        root.geometry("650x550+300+300")
        root.mainloop()
