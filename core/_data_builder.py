import nrrd
import torch
import torchvision.transforms as tf


class DataBuilder:
    def __init__(self,
                 data_path,
                 list_of_categories,
                 num_of_chunks: int = 0,
                 augmentation_coeff: int = 0,
                 num_of_classes: int = 0,
                 normalise: bool = False,
                 fit: bool = True,
                 data_format: int = 0,
                 save_data: bool = False
                 ):
        self.data_path = data_path
        self.number_of_chunks = num_of_chunks
        self.augmentation_coeff = augmentation_coeff
        self.list_of_cats = list_of_categories
        self.num_of_cls = num_of_classes
        self.normalise = normalise
        self.fit = fit
        self.data_format = data_format
        self.save_data = save_data

    def forward(self):
        data = self.get_data()
        data = self.fit_data(data) if self.fit else data
        data = self.pre_normalize(data) if self.normalise else data
        data = self.data_augmentation(data, self.augmentation_coeff) if self.augmentation_coeff != 0 else data
        data = self.new_chunks(data, self.number_of_chunks) if self.number_of_chunks != 0 else data
        data = self.category_splitter(data, self.num_of_cls, self.list_of_cats) if self.num_of_cls != 0 else data
        torch.save(data, self.data_path[-14:]+'.pt') if self.save_data else None

        return torch.unsqueeze(data, 1)

    def get_data(self):
        if self.data_format == 0:
            return torch.from_numpy(nrrd.read(self.data_path)[0])
        elif self.data_format == 1:
            return torch.load(self.data_path).cpu()
        elif self.data_format == 2:
            return torch.unsqueeze(self.data_path, 0).cpu()
        else:
            print('Available types are: "nrrd", "tensor" or "tensor w/o loading"')

    @staticmethod
    def fit_data(some_data):
        data_add_x = torch.nn.ZeroPad2d((0, 0, 8, 0))
        data = data_add_x(some_data)
        data = torch.movedim(data, -1, 0)
        data_add_z = torch.nn.ZeroPad2d((0, 0, 5, 0))
        data = data_add_z(data)
        return torch.movedim(data, 0, -1)  # or -2

    @staticmethod
    def pre_normalize(some_data):
        min_d, max_d = torch.min(some_data), torch.max(some_data)

        return (some_data - min_d) / (max_d - min_d)

    @staticmethod
    def data_augmentation(some_data, aug_n):
        torch.manual_seed(17)
        tr_data = []
        for e in range(aug_n):
            transform = tf.RandomRotation(degrees=(20*e, 20*e))
            for image in some_data:
                image = torch.unsqueeze(image, 0)
                image = transform(image)
                tr_data.append(image)

        return tr_data

    def new_chunks(self, some_data, n_ch):
        data = torch.stack(some_data, 0) if self.augmentation_coeff != 0 else some_data
        data = torch.squeeze(data, 1)
        chunks = torch.chunk(data, n_ch, 0)

        return torch.stack(chunks)

    @staticmethod
    def category_splitter(some_data, alpha, list_of_categories):
        data, _ = torch.squeeze(some_data, 1).to(torch.int64), alpha
        for i in list_of_categories:
            data = torch.where(data < i, _, data)
            _ += 1

        return data - alpha
