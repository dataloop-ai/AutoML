from predictor import predict
import argparse



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--predict", action='store_true', default=False)
        parser.add_argument("--visualize_coco", action='store_true', default=False)
        parser.add_argument("--checkpoint_path", type=str, default='/home/noam/ZazuML/best_checkpoint.pt')
        parser.add_argument("--dataset_path", type=str, default='')
        parser.add_argument("--output_path", type=str, default='')
        args = parser.parse_args()
        if args.predict:
                predict(pred_on_path=args.dataset_path, output_path=args.output_path,
                checkpoint_path=args.checkpoint_path, threshold=0.5)
        if args.visualize_coco:
                from dataloaders import CocoDataset
                dataset = CocoDataset(args.dataset_path, set_name='train')
                dataset.visualize(args.output_path)