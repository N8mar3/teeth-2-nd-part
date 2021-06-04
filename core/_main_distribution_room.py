import torch
import torch.optim as opt
from _Unet_architecture import UNet
from _data_builder import DataBuilder
from _data_loader import get_loaders
from _trainer import TrainFunction


def train_model(
                        data_path,
                        data_num_classes,
                        data_list_of_categories,
                        data_format,
                        data_normalization,
                        data_fit,
                        data_save,
                        mask_path,
                        mask_num_classes,
                        mask_list_of_categories,
                        mask_format,
                        mask_normalization,
                        mask_fit,
                        mask_save,
                        model_name,
                        model_name_to_load,
                        model_input_class,
                        model_output_class,
                        model_num_of_epochs,
                        model_learning_rate,
                        model_number_of_unet_blocks,
                        model_number_of_start_filters,
                        model_activation_function,
                        model_normalization,
                        dims,
                        model_convolution_mode,
                        model_loss_function_mode_binary,
                        model_transfer_learning,
                        num_of_chunks,
                        num_of_workers,
                        batch_size,
                        augmentation_coefficient
                ):
    device = torch.device('cuda:0' if True else 'cpu')
    scale = torch.cuda.amp.GradScaler()
    optimizer = opt.RMSprop
    data = DataBuilder(
                        data_path=data_path,
                        data_format=data_format,
                        list_of_categories=data_list_of_categories,
                        num_of_chunks=num_of_chunks,
                        augmentation_coeff=augmentation_coefficient,
                        num_of_classes=data_num_classes,
                        normalise=data_normalization,
                        fit=data_fit,
                        save_data=data_save,
                        ).forward()

    mask = DataBuilder(
                        data_path=mask_path,
                        data_format=mask_format,
                        list_of_categories=mask_list_of_categories,
                        num_of_chunks=num_of_chunks,
                        augmentation_coeff=augmentation_coefficient,
                        num_of_classes=mask_num_classes,
                        normalise=mask_normalization,
                        fit=mask_fit,
                        save_data=mask_save,
                        ).forward()

    loader = get_loaders(
                        data, mask,
                        batch_size=batch_size,
                        num_workers=num_of_workers
                        )

    model = UNet(
                        in_channels=model_input_class,          out_channels=model_output_class,
                        n_blocks=model_number_of_unet_blocks,   start_filters=model_number_of_start_filters,
                        activation=model_activation_function,   normalization=model_normalization,
                        conv_mode=model_convolution_mode,       dim=dims
                        ).to(device)

    TrainFunction(
                        data_loader=loader,
                        device_for_training=device,
                        model_name=model_name,
                        model_name_pretrained=model_name_to_load,
                        model=model,
                        optimizer=optimizer,
                        learning_rate=model_learning_rate,
                        model_output_class=model_output_class,
                        scale=scale,
                        num_epochs=model_num_of_epochs,
                        transfer_learning=model_transfer_learning,
                        binary_loss_f=model_loss_function_mode_binary
                        ).forward()
