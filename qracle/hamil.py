import pennylane as qml
from pennylane import numpy as np


def generate_h2_hamiltonian(bond_length):
    """
    Generates the Hamiltonian for an H2 molecule with the specified bond length.

    Parameters:
        bond_length (float): Distance between hydrogen atoms (Å).

    Returns:
        H (qml.Hamiltonian): Molecular Hamiltonian.
        qubits (int): Number of qubits required for the Hamiltonian.
        coordinates (np.ndarray): Atomic positions in 3D space.
    """
    # Atomic symbols for H2 molecule
    symbols = ["H", "H"]

    # Hydrogen atoms along the x-axis at ± bond_length/2
    coordinates = np.array([
        [-bond_length / 2, 0.0, 0.0],  # Hydrogen 1
        [bond_length / 2, 0.0, 0.0]    # Hydrogen 2
    ])

    # Generate the Hamiltonian using PennyLane's quantum chemistry module
    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates,
        charge=0, 
        mult=1, 
        basis="sto-3g"
    )

    return H, qubits, coordinates
