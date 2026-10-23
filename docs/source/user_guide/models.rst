======
Models
======

The machine-learned interatomic potentials (MLIPs) known to ML-PEG are listed
below. Each entry is generated from ``ml_peg/models/models.yml``, so it always
matches the configuration ML-PEG loads. Descriptions, and the families models
are grouped into, are defined in ``docs/source/user_guide/model_notes.yml``.

.. note::

   Citations and longer descriptions are still to be added for each model.


uma-s-1p1-omat
--------------

.. code-block:: yaml

   uma-s-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omat"

uma-m-1p1-omat
--------------

.. code-block:: yaml

   uma-m-1p1-omat:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     level_of_theory: PBE
     trained_on_dispersion: false
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omat"

uma-s-1p1-omol
--------------

.. code-block:: yaml

   uma-s-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p1"
       task_name: "omol"

uma-s-1p2-omol
--------------

.. code-block:: yaml

   uma-s-1p2-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-s-1p2"
       task_name: "omol"

uma-m-1p1-omol
--------------

.. code-block:: yaml

   uma-m-1p1-omol:
     module: fairchem.core
     class_name: FAIRChemCalculator
     device: "cpu"
     trained_on_dispersion: true
     level_of_theory: ωB97M-V/def2-TZVPD
     kwargs:
       model_name: "uma-m-1p1"
       task_name: "omol"


MatterSim
=========

mattersim-5M
------------

.. code-block:: yaml

   mattersim-5M:
     module: mattersim.forcefield
     class_name: MatterSimCalculator
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       load_path: "mattersim-v1.0.0-5m"

mattersim-1M
------------

.. code-block:: yaml

   mattersim-1M:
     module: mattersim.forcefield
     class_name: MatterSimCalculator
     device: "cpu"
     trained_on_dispersion: false
     level_of_theory: PBE
     kwargs:
       load_path: "mattersim-v1.0.0-1m"


GRACE
=====

GRACE-2L-OMAT
-------------

.. code-block:: yaml

  grace-2l-omat:
    module: tensorpotential.calculator
    class_name: grace_fm
    device: "cpu"
    trained_on_dispersion: false
    kwargs:
      model: GRACE-2L-OMAT


GRACE-2L-SMAX-OMAT-medium
-------------------------
.. code-block:: yaml

  grace-2l-smax-omat-medium:
    module: tensorpotential.calculator
    class_name: grace_fm
    device: "cpu"
    trained_on_dispersion: false
    kwargs:
      model: GRACE-2L-SMAX-OMAT-medium


SevenNet
========

sevennet-omni-mpa
-----------------

.. code-block:: yaml

  sevennet-omni-mpa:
    module: sevenn.sevennet_calculator
    class_name: SevenNetCalculator
    device: "cpu"
    trained_on_dispersion: false
    level_of_theory: PBE
    kwargs:
      model: "7net-omni"
      modal: "mpa"


ShiftML (NMR shielding models)
==============================

These models predict NMR chemical shieldings, not energies and forces, and are used
only by the :doc:`NMR Models – Organic Materials </user_guide/benchmarks/nmr_organic_materials>`
benchmarks. They are configured in ``ml_peg/models/nmr_models.yml`` rather than
``models.yml``, and installed with the ``shiftml`` extra.

ShiftML3
--------

Trained on periodic GIPAW-PBE shielding tensors. M. Kellner et al., J. Phys. Chem.
Lett. 16, 8714 (2025).

.. code-block:: yaml

  ShiftML3:
    module: shiftml.ase
    class_name: ShiftML
    device: cpu
    level_of_theory: PBE
    kwargs:
      model_version: ShiftML3

ShiftML4
--------

Trained on GIPAW-PBE shielding tensors plus a PBE0-minus-PBE correction computed on
each isolated molecule. M. Kellner et al., arXiv:2608.21313 (2026).

.. code-block:: yaml

  ShiftML4:
    module: shiftml.ase
    class_name: ShiftML
    device: cpu
    level_of_theory: PBE+ΔPBE0(monomer)
    kwargs:
      model_version: ShiftML4
=======
.. model-registry::


ShiftML (NMR shielding models)
==============================

These models predict NMR chemical shieldings, not energies and forces, and are used
only by the :doc:`NMR Models – Organic Materials </user_guide/benchmarks/nmr_organic_materials>`
benchmarks. They are configured in ``ml_peg/models/nmr_models.yml`` rather than
``models.yml``, and installed with the ``shiftml`` extra.

ShiftML3
--------

Trained on periodic GIPAW-PBE shielding tensors. M. Kellner et al., J. Phys. Chem.
Lett. 16, 8714 (2025).

.. code-block:: yaml

  ShiftML3:
    module: shiftml.ase
    class_name: ShiftML
    device: cpu
    level_of_theory: PBE
    kwargs:
      model_version: ShiftML3

ShiftML4
--------

Trained on GIPAW-PBE shielding tensors plus a PBE0-minus-PBE correction computed on
each isolated molecule. M. Kellner et al., arXiv:2608.21313 (2026).

.. code-block:: yaml

  ShiftML4:
    module: shiftml.ase
    class_name: ShiftML
    device: cpu
    level_of_theory: PBE+ΔPBE0(monomer)
    kwargs:
      model_version: ShiftML4
