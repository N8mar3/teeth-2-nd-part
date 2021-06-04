import torch
from vedo import Volume, show, write
from _Unet_architecture import UNet


test_path = 'C:/Coding/Project_13_dcm_teeth_3D_model_builder/' \
            'last_clean/second_part/for_testing_prediction_from_1st_step.pt'
ground_truth = 'C:/Coding/Project_13_dcm_teeth_3D_model_builder/' \
               'last_clean/second_part/for_validating.pt'
project_path = 'C:/Coding/Project_13_dcm_teeth_3D_model_builder/last_clean/second_part/'
model_names = ['models/Unet_new_structure_insanity_1_model.pth',
               'models/Unet_new_structure_insanity_2_model.pth',
               'models/Unet_new_structure_insanity_3_model.pth',
               'models/Unet_new_structure_insanity_4_model.pth',
               'models/Unet_new_structure_insanity_5_model.pth',
               'models/Unet_new_structure_insanity_6_model.pth',
               'models/Unet_new_structure_insanity_7_model.pth',
               'models/Unet_new_structure_insanity_8_model.pth']
prediction = torch.load(test_path).to(dtype=torch.float)  # Prediction from last semantic segmentation


def mask_prepare(some_data):
    mask = torch.load(some_data)
    mask = torch.movedim(mask, 0, -1)

    return mask.cpu().numpy()


def show_save(some_data, save: bool = False, shown: bool = False):
    mask = mask_prepare(ground_truth)
    data_multiclass = Volume(some_data, c='jet', alpha=(0.0, 1), alphaUnit=0.87, mode=1)
    mask_multiclass = Volume(mask, c='jet', alpha=(0.0, 1), alphaUnit=0.87, mode=1)
    data_multiclass.addScalarBar3D()
    mask_multiclass.addScalarBar3D()
    show([(mask_multiclass, "Multiclass teeth segmentation groundtruth"),
          (data_multiclass, "Multiclass teeth segmentation prediction")],
         bg='black', N=2, axes=1).close() if shown else None
    write(data_multiclass.isosurface(), 'multiclass_.obj') if save else None


def predictor(some_data, model, model_name, coefficient):
    data = torch.unsqueeze(some_data, 0)
    data = torch.unsqueeze(data, 0)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    with torch.no_grad():
        predict = model(data)
        predict = torch.squeeze(predict, 0)
        predict = torch.argmax(predict, 0)
        for i in range(1, len(torch.unique(predict))+1):
            predict = torch.where(predict == i, (4*coefficient+i), predict)

    return predict


def model_config(i):
    out_channels = 3 if i == 1 or i == 5 else 5

    model = UNet(in_channels=1,       out_channels=out_channels,
                 n_blocks=3,          start_filters=9,
                 activation='relu',   normalization='batch',
                 conv_mode='same',    dim=3
                 )

    return predictor(prediction[i + 1],
                     model=model,
                     model_name=project_path+model_names[i],
                     coefficient=i+1)


def forward():
    final_prediction = [model_config(i) for i in range(len(model_names))]
    final_prediction = (
                        torch.movedim(
                            torch.sum(
                                torch.stack(
                                    final_prediction),
                                dim=0),
                            0, -1)).numpy()
    show_save(final_prediction, save=False, shown=True)


if __name__ == '__main__':
    forward()
