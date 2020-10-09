from predictor import predict

if __name__ == '__main__':
        predict(pred_on_path='/home/noam/data/tiny_coco/to_predict_on', output_path='/home/noam/data/tiny_coco/predictions',
                checkpoint_path='/home/noam/ZazuML/best_checkpoint.pt', threshold=0.05)
