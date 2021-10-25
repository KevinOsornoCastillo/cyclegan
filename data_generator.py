import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from skimage import io


class DataLoader():

    def __init__(self, params):

        self.norm   = params["norm"]
        self.num_patches = params["num_patches"]
        self.path_data = params["path_data"]

    def random_Shuffle(self, *args):

        indices = list(range(0, len(args[0])))
        np.random.shuffle(indices)
        args = np.asarray(args)
        lista = []

        for i in range(0, len(args)):
            lista.append(args[i,indices])

        return np.asarray(lista)

    def normalize_min_max(self, img):
        img_max = np.max(img)
        img_min = np.min(img)
        img_normalized = (img-img_min)/(img_max-img_min)
        return img_normalized

    def normalize_z_score(self, img):
        img_mean = np.mean(img)
        img_std = np.std(img)
        img_normalized = (img-img_mean)/(img_std)
        return img_normalized

    def comprobacion_len(self, *args):
        for value in args:
            print(len(value))

    def comprobacion_path(self, *args):
        print('\n')
        for value in args:
            print(value)
        print('\n')

    def comprobacion_images(self, a=1, *args):

        if a == 0:
            fig, axs = plt.subplots(int(len(args)/2), 2)
            plt.suptitle('patches', fontsize=20)
            axs[0, 0].set_title('real')
            axs[0, 1].set_title('conditional')
            count = 0
            for i in range(0, int(len(args)/2)):
                for j in range(0, 2):
                    axs[i, j].imshow(args[count], 'gray')
                    count += 1

            plt.show()

        elif a == 1:
            fig, axs = plt.subplots(1, 3)
            plt.suptitle('\n \n \n' + args[3], fontsize=20)
            axs[0].imshow(args[0], 'gray')
            axs[0].set_title('DIN_0')
            axs[1].imshow(args[1], 'gray')
            axs[1].set_title('mask_DIN_0')
            axs[2].imshow(args[2], 'gray')
            axs[2].set_title('Nube de puntos '
                             '#:%d' % len(args[4]))
            plt.show()

        elif a == 2:
            fig, axs = plt.subplots(1, 3)
            plt.suptitle('\n \n \n' + args[3], fontsize=20)
            axs[0].imshow(args[0], 'gray')
            axs[0].set_title('real')
            axs[1].imshow(args[1], 'gray')
            axs[1].set_title('conditional')
            axs[2].imshow(args[2], 'gray')
            axs[2].set_title('DIN_0')
            plt.show()

        elif a == 3:
            fig, axs = plt.subplots(2, 3)
            plt.suptitle(args[6], fontsize=20)
            axs[0, 0].imshow(args[0], 'gray')
            axs[0, 0].set_title('real')
            axs[0, 1].imshow(args[1], 'gray')
            axs[0, 1].set_title('conditional')
            axs[0, 2].imshow(args[2], 'gray')
            axs[0, 2].set_title('DIN_0')
            axs[1, 0].imshow(args[3], 'gray')
            axs[1, 1].imshow(args[4], 'gray')
            axs[1, 2].imshow(args[5], 'gray')
            plt.show()

    def crop_images(self, imgA, imgB, din_0):

        #self.comprobacion_path(imgA, imgB, din_0)

        """ Lectura de imagenes"""
        imgA = io.imread(imgA)
        imgB = io.imread(imgB)
        din_0 = io.imread(din_0)
        din_copy = din_0

        #self.comprobacion_images(2, imgA, imgB, din_0, 'Sin escalar')
        """ Normalizar imagenes """

        if(self.norm == 'min_max'):
            """ Escalado 0-1 """
            imgA = DataLoader.normalize_min_max(self, imgA)
            imgB = DataLoader.normalize_min_max(self, imgB)
            din_0 = DataLoader.normalize_min_max(self, din_0)

        elif(self.norm == 'z_score'):
            """ Normalización Z-Score """
            imgA = DataLoader.normalize_z_score(self, imgA)
            imgB = DataLoader.normalize_z_score(self, imgB)
            din_0 = DataLoader.normalize_z_score(self, din_0)

        #self.comprobacion_images(2, imgA, imgB, din_0, 'Escalada 0 - 1')

        """ Se calculan los centros para cortar los patches """
        row, columns = din_0.shape[:2]
        n_points = 1000
        mask_din_0 = din_0 > 0.1

        p_32 = p = 32

        xc = np.random.randint(p, row - p - 1, n_points)
        yc = np.random.randint(p, columns - p - 1, n_points)

        xc_new = np.array([], dtype=int)
        yc_new = np.array([], dtype=int)

        for j in range(0, len(xc)):
            if mask_din_0[xc[j], yc[j]] == True:
                xc_new = np.append(xc_new, xc[j])
                yc_new = np.append(yc_new, (yc[j]))

        """ Se crea una mascara nueva para mostrar la nube de puntos """
        mask = np.zeros((row, columns))
        mask[xc_new, yc_new] = 1

        #self.comprobacion_images(1, din_0, mask_din_0, mask, 'Nube de puntos', xc_new )

        """ Con los centros calculados, se corta cada uno de los patches de la imagen """
        crop_A = []
        crop_32 = []



        #mascara_A = color.gray2rgb(imgA)
        #mascara_B = color.gray2rgb(imgB)
        #mascara_din = color.gray2rgb(din_0)

        for k in range(0, len(xc_new)):

            t_crop_A = imgA[int(xc_new[k] - p_32/2): int((xc_new[k] + p_32/2)), int(yc_new[k] - p_32/2): int(yc_new[k] + p_32/2)]
            t_crop_32 = imgB[int(xc_new[k] - p_32/2): int((xc_new[k] + p_32/2)), int(yc_new[k] - p_32/2): int(yc_new[k] + p_32/2)]


            din_0_corte = din_copy[int(xc_new[k] - p_32/2): int((xc_new[k] + p_32/2)), int(yc_new[k] - p_32/2): int(yc_new[k] + p_32/2)]

            crop_A.append(t_crop_A)
            crop_32.append(t_crop_32)


            # """ Para mostrar el patche cortado """
            # mascara_A = color.gray2rgb(imgA)
            # mascara_B = color.gray2rgb(imgB)
            # mascara_din = color.gray2rgb(din_0)
            # p = p_32
            # mascara_A[int(xc_new[k] - p/2): int((xc_new[k] + p/2)), int(yc_new[k] - p/2): int(yc_new[k] + p/2), 0] = 1
            # mascara_B[int(xc_new[k] - p/2): int((xc_new[k] + p/2)), int(yc_new[k] - p/2): int(yc_new[k] + p/2), 0] = 1
            # mascara_din[int(xc_new[k] - p/2): int((xc_new[k] + p/2)), int(yc_new[k] - p/2): int(yc_new[k] + p/2), 0] = 1
            # self.comprobacion_images(3, mascara_A, mascara_B, mascara_din, din_0_corte, din_0_corte, din_0_corte, 'Crop images')

        #self.comprobacion_images(3, mascara_din, mascara_B, mascara_din, t_crop_A, t_crop_32, din_0_corte, 'Crop images')

        crop_A = crop_A[0:self.num_patches]
        crop_32 = crop_32[0:self.num_patches]

        return [crop_32, crop_A]

    def carga_de_imagenes(self, batch_size=32, Shuffle=True, Training=True, n_images=3):


        self.training=Training
        
        """ Se carga la ruta donde estan almacenadas las imagenes """
        if Training == True:
            ImageA= np.genfromtxt('/home/kevin/Drive/Research/2021/1.Proyecto_Jovenes_Talento/03-Modelos-Computacionales/01-CycleGAN/_Data/d2/train/train.csv', dtype=str, delimiter=',')
            ImageB= np.genfromtxt('/home/kevin/Drive/Research/2021/1.Proyecto_Jovenes_Talento/03-Modelos-Computacionales/01-CycleGAN/_Data/d0/train/train.csv', dtype=str, delimiter=',')
            Dinamico_0= np.genfromtxt('/home/kevin/Drive/Research/2021/1.Proyecto_Jovenes_Talento/03-Modelos-Computacionales/01-CycleGAN/_Data/d0/train/train.csv', dtype=str, delimiter=',')
            ImageA = ImageA[1:]
            ImageB = ImageB[1:]
            Dinamico_0 = Dinamico_0[1:]

        elif Training == False:
            ImageA = np.loadtxt(f'{self.path_data}/din2/test.csv', dtype=str, delimiter=';')
            ImageB = np.loadtxt(f'{self.path_data}/din0/test.csv', dtype=str, delimiter=';')
            Dinamico_0 = np.loadtxt(f'{self.path_data}/din0/test.csv', dtype=str, delimiter=';')

        #self.comprobacion_len(ImageA, ImageB, Dinamico_0)
        

        """ Se ordenan las imagenes de forma aleatoria """
        if Shuffle == True:
            ImageA, ImageB, Dinamico_0 = self.random_Shuffle(ImageA, ImageB, Dinamico_0)

        """ Se abren las imagenes en lotes """
        for i in range(0, len(ImageA) - n_images, n_images):


            crop_Img32 = []
            crop_ImgA = []

            for l in range(0, n_images):
                a = self.crop_images(ImageA[i+l], ImageB[i+l], Dinamico_0[i+l])
                crop_Img32.append(a[0])
                crop_ImgA.append(a[1])

            crop_Img32 = [item for l in crop_Img32 for item in l]
            crop_ImgA = [item for l in crop_ImgA for item in l]


            self.n_subbatches = int(len(crop_ImgA) / batch_size)

            for z in range(self.n_subbatches):
                Patch_32 = crop_Img32[z * batch_size:(z + 1) * batch_size]
                Patch_A = crop_ImgA[z*batch_size:(z+1)*batch_size]

                yield Patch_32, Patch_A




if __name__ == '__main__':

    data_loader_params = {"is_gpu": False,
                          "norm": "z_score",
                          "num_patches": 100,
                          "path_data": "/home/kevin/Drive/Research/2021/1.Proyecto_Jovenes_Talento/03-Modelos-Computacionales/01-CycleGAN/_Data"}

    dataloader = DataLoader(data_loader_params)
    for i, (train_A, train_B) in enumerate(dataloader.carga_de_imagenes
                                               (batch_size=1)):
        print(dataloader.n_subbatches)
        print(i) 
