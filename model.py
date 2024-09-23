import torchvision


def create_model(
        num_classes, # including background class
        trainable_backbone_layers=None,
        pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=trainable_backbone_layers
    )

    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_model(2)

    #print(model)
    print(model.transform.max_size)