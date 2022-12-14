from .json_dataset import JsonDataset

DATASETS = {
    JsonDataset.code(): JsonDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
