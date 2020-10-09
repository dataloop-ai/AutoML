import torch
from .pred_utils import detect
from logging_utils import init_logging, logginger
logger = logginger(__name__)

def load_inference(checkpoint_path):
    if torch.cuda.is_available():
        logger.info('cuda available')
        return torch.load(checkpoint_path)
    else:
        logger.info('run on cpu')
        return torch.load(checkpoint_path, map_location=torch.device('cpu'))


def predict(pred_on_path, output_path, checkpoint_obj=None, checkpoint_path='checkpoint.pt', threshold=0.5):

    if checkpoint_obj is None:
        checkpoint_obj = load_inference(checkpoint_path)
    return detect(checkpoint=checkpoint_obj, pred_on_path=pred_on_path, output_path=output_path, threshold=threshold, visualize=True)
