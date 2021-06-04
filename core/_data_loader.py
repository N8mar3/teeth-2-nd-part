import torch.utils.data as tud


class ToothDataset(tud.Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self): return len(self.images)

    def __getitem__(self, index):
        if self.masks is not None:
            return self.images[index, :, :, :],\
                    self.masks[index, :, :, :]
        else:
            return self.images[index, :, :, :]


def get_loaders(images, masks,
                batch_size: int = 1,
                num_workers: int = 1,
                pin_memory: bool = True):
    train_ds = ToothDataset(images=images,
                            masks=masks)

    data_loader = tud.DataLoader(train_ds,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    return data_loader
