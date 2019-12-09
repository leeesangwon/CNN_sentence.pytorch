from .cnn import CNN, CNNMultiChannel


MODELS = ('random', 'static', 'non-static', 'multi-channel')


def get_model(model_type, num_classes, pretrained_word2vec):
    if model_type not in MODELS:
        raise ValueError("Invalid model type: %s" % model_type)

    if model_type == 'random':
        return CNN(num_classes, pretrained_word2vec, False, False)
    elif model_type == 'static':
        return CNN(num_classes, pretrained_word2vec, True, True)
    elif model_type == 'non-static':
        return CNN(num_classes, pretrained_word2vec, True, False)
    else:
        return CNNMultiChannel(num_classes, pretrained_word2vec)
