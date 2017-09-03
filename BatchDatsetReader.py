"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import gdal
import read_MITSceneParsingData as scene_parsing

class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    ImageSize=224
    def __init__(self, records_list, image_options={}):
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()
        self.ImageSize=image_options["resize_size"]

    def _read_images(self):
        # self.__channels = True
        # self.images = np.array([self._transform(filename['image'],True) for filename in self.files])
        self.images = np.array([filename['image'] for filename in self.files])
        # self.__channels = False
        # self.annotations = np.array(
        #     [np.expand_dims(self._transform(filename['annotation'],False), axis=3) for filename in self.files])
        self.annotations = np.array(
            [filename['annotation'] for filename in self.files])
        # print (self.images.shape)
        # print (self.annotations.shape)

    def _transform(self, filename,flag):
        if flag:
            image=gdal.Open(filename).ReadAsArray(0,0,self.ImageSize,self.ImageSize)
            image_copy=np.zeros([self.ImageSize,self.ImageSize,3])
            for s in range(len(image)):
                image_copy[:,:,s]=image[s]
            image=image_copy
            resize_image = image
        else:
            image=np.zeros([self.ImageSize, self.ImageSize])
            with open(filename) as f:
                i=0
                for line in f:
                    content=np.array(map(float,line.strip().split()))
                    content=map(int, content)
                    image[i]=content
                    i+=1
            resize_image=image
        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.images):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        # return self.images[start:end], self.annotations[start:end]
        data=np.array([self._transform(filename,True) for filename in self.images[start:end]])
        labels=np.array([np.expand_dims(self._transform(filename,False), axis=3) 
            for filename in self.annotations[start:end]])
        return data,labels
        
    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, int(len(self.images)), size=[batch_size]).tolist()
        data=np.array([self._transform(filename,True) for filename in self.images[indexes]])
        labels=np.array([np.expand_dims(self._transform(filename,False), axis=3) 
            for filename in self.annotations[indexes]])
        return data, labels

