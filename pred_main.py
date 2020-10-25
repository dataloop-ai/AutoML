from predictor import predict
import argparse



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--predict", action='store_true', default=False)
        parser.add_argument("--visualize_coco", action='store_true', default=False)
        parser.add_argument("--dataset_path", type=str, default='')
        parser.add_argument("--output_path", type=str, default='')
        args = parser.parse_args()
        if args.predict:
                predict(pred_on_path=args.dataset_path, output_path=args.output_path,
                checkpoint_path='/home/noam/ZazuML/best_checkpoint.pt', threshold=0.05)
        if args.visualize_coco:
                from dataloaders import CocoDataset
                dataset = CocoDataset(args.dataset_path, set_name='train')
                dataset.visualize(args.output_path)