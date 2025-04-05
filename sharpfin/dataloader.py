import msgspec
import math
import warnings
import os
from pathlib import Path
from typing import Type, TypeVar, List, Optional, Generic, Callable, Dict, Tuple, Iterator, Any
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torchvision.transforms.v2 as v2
from  . import transforms as SFT
import numpy as np
from collections import namedtuple, defaultdict
from tqdm.auto import tqdm

Bucket = namedtuple('Bucket', ['height', 'width'])

# Generic type variable for conditioning
T = TypeVar("T", bound=msgspec.Struct)

class ImageDatasetItem(msgspec.Struct, Generic[T]):
    image_path: str
    height: Optional[int] = None # optional, can be read from images (but slow on large datasets)
    width: Optional[int] = None
    # Generic for user-defined conditioning structure.
    # Dataloader lets you define your own processing function for this.
    conditioning: Optional[T] = None

    # attributes that generally will be processed by the dataloader but which can be reused to skip processing
    _buckets: Optional[List[Bucket]] = []

    @property
    def log_aspect_ratio(self) -> Optional[float]:
        if self.width and self.height:
            return math.log(self.width / self.height)
        return None

class BucketedDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        conditioning_cls: Type[T],
        conditioning_processor: Callable[[T, torch.Generator], Dict] = lambda x: [x],
        rng: torch.Generator = torch.Generator('cpu'),
        data_base_path: Optional[str] = None,
        bucket_sizes: Optional[List[int]] = None,
        bucket_max_ar: float = 2.0,
        bucket_step: int = 64,
        bucket_groups: Optional[dict[List[Bucket]]] = None,
        save_processed: bool = True,
        processed_json_path: Optional[str] = None,
    ):
        self.json_path = Path(json_path)
        self.data_base_path = Path(data_base_path)
        self.processed_json_path = (
            Path(processed_json_path) if processed_json_path else self.json_path.with_suffix(".processed.json")
        )
        self.save_processed = save_processed
        self.conditioning_cls = conditioning_cls
        self.conditioning_processor = conditioning_processor
        self.rng = rng

        if bucket_groups is not None:
            self.bucket_groups = bucket_groups
        elif bucket_sizes is not None:
            self.bucket_groups = {res: _bucket_generator(res, bucket_max_ar, bucket_step) for res in bucket_sizes}
        else:
            raise ValueError("Must pass either a list of bucket groups [as a list of lists of Buckets] or a list of base resolutions to construct buckets for.")
        
        self._bucket_aspect_ratios = {}
        self._bucket_item_indices = defaultdict(list)

        # Through some miracle, this does not seem to suffer from the same refcounting issues
        # that most torch dataloaders suffer from when using python lists. This must be studied.
        self.dataset = self._load_and_process_dataset()

    def _load_and_process_dataset(self) -> List[ImageDatasetItem[T]]:
        """Loads the dataset, verifies fields, and assigns bucket IDs."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Dataset JSON file not found: {self.json_path}")

        # Load JSON data using msgspec
        print("Reading dataset json...")
        with open(self.json_path, "rb") as f:
            raw_data = f.read()

        # Deserialize items
        print("Deserializing dataset json...")
        dataset = msgspec.json.decode(raw_data, type=List[ImageDatasetItem[self.conditioning_cls]])

        updated = False  # Track if we modified the dataset
        needs_bucketing = True

        # Process missing fields
        for item in tqdm(dataset, desc="Checking dataset attributes..."):
            if item.width is None or item.height is None:
                warnings.warn(f"Missing dimensions for {item.image_path}, reading from disk (SLOW).")
                try:
                    with Image.open(item.image_path) as img:
                        item.width, item.height = img.size
                    updated = True
                except Exception as e:
                    warnings.warn(f"Failed to open {item.image_path}: {e}")
            if len(item._buckets) > 0:
                # Assume we need to bucket unless at least one image has buckets assigned already.
                needs_bucketing = False

        # Assign all images to buckets
        if needs_bucketing:
            self._bucket_assignment(dataset)
            updated = True

        # Build a hash table so we easily can get all images in a bucket
        self._bucket_hash_table(dataset)

        # Save processed dataset if required
        if updated and self.save_processed and self.processed_json_path:
            with open(self.processed_json_path, "wb") as f:
                f.write(msgspec.json.encode(dataset))
            print(f"Processed dataset saved to {self.processed_json_path}.")
            print(f"If running again with the same bucketing settings, you can save processing time by loading this directly.")

        # TODO: Maybe delete the dataset _bucket keys since they won't be used past building HT?

        return dataset

    def _bucket_assignment(
            self,
            dataset: List[ImageDatasetItem],
            strategy: str = "nearest",
            allow_upscale: bool = False
        ):
        for base_res in self.bucket_groups.keys():
            if self._bucket_aspect_ratios.get(base_res) is None:
                self._bucket_aspect_ratios[base_res] = np.array([math.log(b.width/b.height) for b in self.bucket_groups[base_res]])

            group_aspect_ratios = self._bucket_aspect_ratios[base_res]
            bucket_group = self.bucket_groups[base_res]

            for idx, item in enumerate(tqdm(dataset, desc=f"Assigning buckets for resolution {base_res}")):
                match strategy:
                    case "nearest":
                        nearest_bucket = bucket_group[np.argmin(np.abs(group_aspect_ratios - item.log_aspect_ratio))]

                    case "crop_long":
                        bucket_candidates = group_aspect_ratios - item.log_aspect_ratio
                        np.where(bucket_candidates < 0, 999, bucket_candidates)
                        nearest_bucket = bucket_group[np.argmin(np.where(bucket_candidates < 0, 999, bucket_candidates))]
                    case _:
                        raise ValueError(f"Unknown bucket assignment strategy {strategy}")

                if (nearest_bucket.width <= item.width and nearest_bucket.height <= item.height) or allow_upscale:
                    item._buckets.append(nearest_bucket)

    def _bucket_hash_table(
            self,
            dataset: List[ImageDatasetItem]
        ):
        for idx, item in enumerate(tqdm(dataset, desc="Building bucket hash table...")):
            for bucket in item._buckets:
                self._bucket_item_indices[bucket].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Tuple[int, Bucket]) -> Dict[str, Any]:
        item_idx, bucket = idx
        transform = v2.Compose([
            SFT.ApplyCMS(),
            SFT.AlphaComposite(),
            SFT.AspectRatioCrop(bucket.width, bucket.height),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            SFT.Scale(
                (bucket.height, bucket.width),
                out_dtype=torch.float16,
            )
        ])
        if self.data_base_path is not None:
            image_path = os.path.join(self.data_base_path, self.dataset[item_idx].image_path)
        else:
            image_path = self.dataset[item_idx].image_path

        output = {}
        output['image'] = transform(Image.open(image_path))
        output.update(self.conditioning_processor(self.dataset[item_idx].conditioning, self.rng))
        return output

class BucketedBatchSampler(Sampler[List[Tuple[int, Bucket]]]):
    def __init__(self, bucket_dict: Dict[Bucket, List[int]], batch_size: int, rng: np.random.Generator = None):
        """
        Args:
            bucket_dict: A dictionary where keys are Buckets and values are lists of indices.
            batch_size: Number of samples per batch.
            rng: Optional numpy random generator for reproducibility.
        """
        self.rng = rng if rng else np.random.default_rng()
        self.bucket_keys = list(bucket_dict.keys())
        self.buckets = [np.array(bucket_dict[key], dtype=int) for key in self.bucket_keys]
        self.progress = np.zeros(len(self.buckets), dtype=int)
        self.batch_size = batch_size
        self.batch_order = []
        self.shuffle()

    def shuffle(self):
        """Shuffles the dataset and determines the batch order."""
        for bucket in self.buckets:
            self.rng.shuffle(bucket)
        
        batch_counts = [len(bucket) // self.batch_size for bucket in self.buckets]
        self.batch_order = self.rng.choice(
            len(self.buckets), sum(batch_counts), p=np.array(batch_counts) / sum(batch_counts)
        )
        self.progress.fill(0)

    def __iter__(self) -> Iterator[List[Tuple[int, Bucket]]]:
        """Yields batches of (index, Bucket) tuples."""
        for bucket_idx in self.batch_order:
            bucket = self.buckets[bucket_idx]
            start_idx = self.progress[bucket_idx]
            end_idx = start_idx + self.batch_size
            self.progress[bucket_idx] = end_idx
            yield [(idx, self.bucket_keys[bucket_idx]) for idx in bucket[start_idx:end_idx]]

    def __len__(self) -> int:
        """Returns the number of batches."""
        return len(self.batch_order)

# adapted from https://github.com/lodestone-rock/flow/blob/master/src/dataloaders/bucketing_logic.py
def _bucket_generator(base_resolution=256, ratio_cutoff=2, step=64):
    # bucketing follows the logic of 1/x
    # https://www.desmos.com/calculator/byizruhsry
    x = base_resolution
    y = base_resolution
    half_buckets = [(x, y)]
    while y / x <= ratio_cutoff:
        y += step
        x = int((base_resolution**2 / y) // step * step)
        if x != half_buckets[-1][0]:
            half_buckets.append(
                (
                    x,
                    y,
                )
            )

    another_half_buckets = []
    for bucket in reversed(half_buckets[1:]):
        another_half_buckets.append(bucket[::-1])

    return [Bucket(width=w, height=h) for w, h in another_half_buckets + half_buckets]