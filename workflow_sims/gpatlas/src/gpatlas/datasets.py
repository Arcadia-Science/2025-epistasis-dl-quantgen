import h5py
import torch
from torch.utils.data import Dataset
from typing import cast, List, Tuple
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self, hdf5_path: Path) -> None:
        self.hdf5_path = hdf5_path
        self.h5 = None
        self._strain_group = None
        self.strains = None
        # Open temporarily to get keys and length for initialization
        with h5py.File(self.hdf5_path, "r") as temp_h5:
            temp_strain_group = cast(h5py.Group, temp_h5["strains"])
            self._strain_keys: List[str] = list(temp_strain_group.keys())
            self._len = len(temp_strain_group)

    def _init_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.hdf5_path, "r")
            self._strain_group = cast(h5py.Group, self.h5["strains"])
            self.strains = self._strain_keys

    def __len__(self) -> int:
        return self._len


class GenoPhenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._init_h5()
        strain = self.strains[idx]
        strain_data = cast(Dataset, self._strain_group[strain])

        phens = torch.tensor(strain_data["phenotype"][:], dtype=torch.float32)
        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()

        return phens, gens


class PhenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> torch.Tensor:
        self._init_h5()
        strain = self.strains[idx]
        strain_data = cast(Dataset, self._strain_group[strain])

        phens = torch.tensor(strain_data["phenotype"][:], dtype=torch.float32)

        return phens


class GenoDataset(BaseDataset):
    def __getitem__(self, idx: int) -> torch.Tensor:
        self._init_h5()
        strain = self.strains[idx]
        strain_data = cast(Dataset, self._strain_group[strain])

        gens = torch.tensor(strain_data["genotype"][:], dtype=torch.float32).flatten()

        return gens


def create_data_loaders(base_file_name, batch_size=128, num_workers=3, shuffle=True):
    train_data_geno = GenoDataset(Path(f'{base_file_name}train.hdf5'))
    test_data_geno = GenoDataset(Path(f'{base_file_name}test.hdf5'))

    train_data_gp = GenoPhenoDataset(Path(f'{base_file_name}train.hdf5'))
    test_data_gp = GenoPhenoDataset(Path(f'{base_file_name}test.hdf5'))

    train_data_pheno = PhenoDataset(Path(f'{base_file_name}train.hdf5'))
    test_data_pheno = PhenoDataset(Path(f'{base_file_name}test.hdf5'))

    train_loader_geno = torch.utils.data.DataLoader(
        dataset=train_data_geno, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle, pin_memory=False
    )
    test_loader_geno = torch.utils.data.DataLoader(
        dataset=test_data_geno, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )

    train_loader_pheno = torch.utils.data.DataLoader(
        dataset=train_data_pheno, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )
    test_loader_pheno = torch.utils.data.DataLoader(
        dataset=test_data_pheno, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )

    train_loader_gp = torch.utils.data.DataLoader(
        dataset=train_data_gp, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )
    test_loader_gp = torch.utils.data.DataLoader(
        dataset=test_data_gp, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle
    )

    return {
        'train_loader_geno': train_loader_geno,
        'test_loader_geno': test_loader_geno,
        'train_loader_pheno': train_loader_pheno,
        'test_loader_pheno': test_loader_pheno,
        'train_loader_gp': train_loader_gp,
        'test_loader_gp': test_loader_gp,
    }
