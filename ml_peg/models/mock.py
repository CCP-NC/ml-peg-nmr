"""Mock calculators (MLIPs) and mock shielding predictors (NMR models)."""

from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import numpy as np


class MockCalculator(Calculator):
    """A mock calculator that returns zero values for all properties."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
        **kwargs,
    ) -> None:
        """
        Define calculation method for mock calculator.

        Parameters
        ----------
        atoms
            Atoms object to calculate properties for.
        properties
            List of properties to calculate.
        system_changes
            List of system changes to consider for calculation.
        **kwargs
            Any additional keyword arguments.
        """
        super().calculate(atoms, properties, system_changes)

        if "energy" in properties:
            self.results["energy"] = 0.0

        if "forces" in properties:
            self.results["forces"] = np.zeros((len(self.atoms), 3))

        if "stress" in properties:
            self.results["stress"] = np.zeros(6)


class MockErrorCalculator(Calculator):
    """A mock calculator that raises an error for all properties."""

    implemented_properties = ["energy", "forces", "stress"]
    results = {}
    parameters = {}

    def calculate(self, *args, **kwargs) -> None:
        """
        Define calculation method for mock calculator.

        Parameters
        ----------
        *args
            Any additional positional arguments.
        **kwargs
            Any additional keyword arguments.
        """
        raise ValueError("This is a mock error calculator. All calculations fail.")


class MockShieldingPredictor:
    """
    Magnetic shielding counterpart of ml-peg's ``MockCalculator``.

    Every shielding value is zero. Run by default alongside the models, as the mock
    calculator is for the MLIP benchmarks, so that every benchmark always has one
    complete set of outputs.
    """

    def get_cs_iso_ensemble(self, atoms: Atoms) -> np.ndarray:
        """
        Return zero isotropic shieldings for a one-member ensemble.

        Parameters
        ----------
        atoms
            Structure to predict.

        Returns
        -------
        np.ndarray
            Zeros, shape (atoms, 1).
        """
        return np.zeros((len(atoms), 1))

    def get_cs_tensor_ensemble(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """
        Return zero shielding tensors for a one-member ensemble.

        Parameters
        ----------
        atoms
            Structure to predict.
        **kwargs
            Ignored (e.g. ``return_symmetric``).

        Returns
        -------
        np.ndarray
            Zeros, shape (atoms, 3, 3, 1).
        """
        return np.zeros((len(atoms), 3, 3, 1))


class MockErrorShieldingPredictor:
    """
    Magnetic shielding counterpart of ml-peg's ``MockErrorCalculator``.

    Every call fails.
    """

    def get_cs_iso_ensemble(self, atoms: Atoms) -> np.ndarray:
        """
        Fail, as ml-peg's error calculator does.

        Parameters
        ----------
        atoms
            Structure to predict.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError("This is a mock error predictor. All predictions fail.")

    def get_cs_tensor_ensemble(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """
        Fail, as ml-peg's error calculator does.

        Parameters
        ----------
        atoms
            Structure to predict.
        **kwargs
            Ignored.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError("This is a mock error predictor. All predictions fail.")
